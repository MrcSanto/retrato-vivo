"""Testes da correção de perspectiva ("foto de uma foto" tirada com o celular)."""

import os

import cv2
import numpy as np
import pytest

from retrato_vivo import perspective as P
from retrato_vivo.cli import build_parser
from retrato_vivo.pipeline import STEP_INFO, RestorationConfig, restore

from test_restauracao import make_scene

PAPER_QUAD = np.float32([[230, 140], [1010, 190], [960, 760], [190, 700]])   # sup.-esq., sup.-dir., inf.-dir., inf.-esq.


def phone_photo(background=(40, 45, 55), quad=PAPER_QUAD, size=(900, 1200), seed=0):
    """Foto impressa (cena com margem branca) fotografada em perspectiva sobre uma mesa."""
    rng = np.random.default_rng(seed)
    h, w = size
    paper = cv2.copyMakeBorder(make_scene(450, 600, seed), 20, 20, 20, 20, cv2.BORDER_CONSTANT,
                               value=(235, 238, 240))
    ph, pw = paper.shape[:2]
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]), quad)
    table = np.full((h, w, 3), background, np.float32) + rng.normal(0, 4, (h, w, 3))
    inside = cv2.warpPerspective(np.ones((ph, pw), np.float32), matrix, (w, h))[..., None]
    img = table * (1 - inside) + cv2.warpPerspective(paper.astype(np.float32), matrix, (w, h)) * inside
    return np.clip(cv2.GaussianBlur(img, (0, 0), 1.0), 0, 255).astype(np.uint8)


def test_ordenar_cantos_in_any_order():
    shuffled = PAPER_QUAD[[2, 0, 3, 1]]
    assert np.array_equal(P.ordenar_cantos(shuffled), PAPER_QUAD)


@pytest.mark.parametrize("background", [(40, 45, 55), (60, 100, 140)])     # mesa escura, madeira
def test_finds_the_paper_corners(background):
    corners = P.encontrar_papel(phone_photo(background))
    assert corners is not None
    assert np.abs(corners - PAPER_QUAD).max() < 12          # poucos pixels, numa foto de 1200 px


def test_rectified_photo_is_straight_and_without_table():
    out, found = P.corrigir_perspectiva(phone_photo())
    assert found
    h, w = out.shape[:2]
    assert 1.15 < w / h < 1.55                              # proporção do papel (640x490 = 1,31)
    # nada da mesa escura nas bordas: a margem branca do papel vai de ponta a ponta
    for strip in (out[:4], out[-4:], out[:, :4], out[:, -4:]):
        assert strip.mean() > 180


def test_scanned_photo_comes_back_untouched():
    img = make_scene(600, 800)
    out, found = P.corrigir_perspectiva(img)
    assert not found and out is img


def test_paper_that_fills_the_whole_photo_is_left_alone():
    """Sem fundo em volta não há o que recortar: provavelmente é uma foto digitalizada com moldura."""
    framed = cv2.copyMakeBorder(make_scene(560, 760), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(235, 235, 235))
    assert P.encontrar_papel(framed) is None


def test_desenhar_cantos_does_not_change_the_input():
    img = phone_photo()
    before = img.copy()
    drawn = P.desenhar_cantos(img, PAPER_QUAD)
    assert np.array_equal(img, before) and not np.array_equal(drawn, img)


# ================================================================ pipeline

def test_pipeline_leaves_perspective_off_by_default():
    img = phone_photo()
    res = restore(img)
    assert res.original.shape == img.shape and "perspectiva" not in res.steps


def test_pipeline_corrects_perspective_when_asked():
    img = phone_photo()
    res = restore(img, RestorationConfig(correct_perspective=True))
    assert "papel_detectado" in res.steps and "perspectiva" in res.steps
    assert res.original.shape == res.steps["perspectiva"].shape        # o "antes" já é a foto desentortada
    assert res.image.shape == res.original.shape
    assert res.original.shape[1] < img.shape[1]
    assert "desentortado" in res.notes["perspectiva"]


def test_pipeline_explains_when_there_is_no_paper():
    res = restore(make_scene(400, 600), RestorationConfig(correct_perspective=True))
    assert "perspectiva" not in res.steps
    assert res.notes["perspectiva"].startswith("pulada")
    assert res.original.shape == (400, 600, 3)


def test_mask_from_the_editor_fits_the_corrected_photo():
    """A máscara pintada na GUI tem o tamanho de ``result.original``: a 2ª restauração tem que aceitá-la."""
    cfg = RestorationConfig(correct_perspective=True)
    first = restore(phone_photo(), cfg)
    mask = np.zeros(first.original.shape[:2], np.uint8)
    mask[100:120, 100:300] = 255
    second = restore(phone_photo(), cfg, mask_add=mask)
    assert second.damage.mask[110, 200] > 0


def test_perspective_steps_have_descriptions():
    assert "papel_detectado" in STEP_INFO and "perspectiva" in STEP_INFO


def test_cli_accepts_perspective_option():
    assert build_parser().parse_args(["restaurar", "f.jpg", "--perspectiva"]).perspectiva
    assert not build_parser().parse_args(["restaurar", "f.jpg"]).perspectiva


def test_gui_checkbox_turns_perspective_on():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from retrato_vivo.gui.controls import ControlsPanel
    QApplication.instance() or QApplication([])
    panel = ControlsPanel()
    assert not panel.config().correct_perspective
    panel.correct_perspective.setChecked(True)
    assert panel.config().correct_perspective
