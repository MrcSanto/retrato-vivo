"""Testes do diferencial: Máquina do tempo e Retrato Vivo (e as abas da interface)."""

import os

import cv2
import numpy as np
import pytest

from retrato_vivo import living_portrait as lp
from retrato_vivo.time_machine import ERAS, travel

from test_restauracao import make_scene

needs_depth = pytest.mark.skipif(not lp.is_available(), reason="modelo de profundidade não baixado")


# ========================================================== Máquina do tempo

@pytest.mark.parametrize("era", list(ERAS))
def test_every_era_returns_a_photo(era):
    out = travel(make_scene(300, 400), era, seed=1)
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3
    assert min(out.shape[:2]) >= 300              # moldura nunca diminui a foto


def test_same_seed_same_result():
    img = make_scene(200, 300)
    assert np.array_equal(travel(img, "1920", seed=7), travel(img, "1920", seed=7))
    assert not np.array_equal(travel(img, "1920", seed=7), travel(img, "1920", seed=8))


def test_black_and_white_eras_are_gray_and_polaroid_is_square_with_wide_bottom():
    img = make_scene(300, 400)
    out = travel(img, "1950", seed=1)
    inner = out[30:-30, 30:-30].astype(int)
    assert np.abs(inner[..., 0] - inner[..., 2]).max() <= 2      # P&B
    polaroid = travel(img, "1990", seed=1)
    assert polaroid.shape[0] > polaroid.shape[1]                 # base mais larga que o topo


def test_unknown_era_is_rejected():
    with pytest.raises(ValueError):
        travel(make_scene(50, 50), "2077")


# =============================================================== Retrato Vivo

def test_split_layers_keeps_the_near_object():
    depth = np.full((200, 300), 0.2, np.float32)
    cv2.circle(depth, (150, 110), 60, 0.9, -1)                   # um objeto bem perto
    alpha = lp.split_layers(depth)
    assert alpha[110, 150] > 0.9 and alpha[10, 10] < 0.1


@needs_depth
def test_depth_is_normalized_and_has_photo_size():
    depth = lp.estimate_depth(make_scene(240, 320))
    assert depth.shape == (240, 320) and depth.dtype == np.float32
    assert 0.0 <= depth.min() and depth.max() <= 1.0


@needs_depth
def test_living_portrait_moves_and_saves(tmp_path):
    cfg = lp.LivingPortraitConfig(seconds=1.0, fps=10, max_side=320)
    portrait = lp.make_living_portrait(make_scene(241, 321), cfg)
    assert len(portrait.frames) == 10
    h, w = portrait.frames[0].shape[:2]
    assert h % 2 == 0 and w % 2 == 0                              # MP4 exige lados pares
    assert not np.array_equal(portrait.frames[0], portrait.frames[3])  # a câmera se mexe
    video = lp.save_video(portrait, tmp_path / "vídeo_ção")         # caminho com acentos
    gif = lp.save_gif(portrait, tmp_path / "anim")
    assert video.suffix == ".mp4" and video.stat().st_size > 1000
    assert gif.suffix == ".gif" and gif.stat().st_size > 1000


# ================================================================= interface

@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_extra_tabs_exist_and_wait_for_a_photo(app):
    from retrato_vivo.gui.main_window import MainWindow
    window = MainWindow()
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert "Retrato Vivo" in titles and "Máquina do tempo" in titles
    assert not window.living_view.create_button.isEnabled()
    assert not any(b.isEnabled() for b in window.time_machine.era_buttons.values())
    window.time_machine.set_source(make_scene(100, 150))
    assert all(b.isEnabled() for b in window.time_machine.era_buttons.values())
    window.time_machine.travel_to("1920")
    assert window.time_machine.shown is not None and window.time_machine.restore_back.isEnabled()
    window.close()


def test_advanced_settings_start_hidden(app):
    from retrato_vivo.gui.controls import ControlsPanel
    panel = ControlsPanel()
    assert not panel.advanced.isVisibleTo(panel)
    panel.advanced_toggle.setChecked(True)
    assert panel.advanced.isVisibleTo(panel)
