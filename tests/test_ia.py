"""Testes das partes de IA pesada: runtime (GPU/CPU), LaMa, detector de riscos,
restauração de rostos e DDColor.

Os testes que precisam de um modelo são pulados se ele não foi baixado
(``python -m retrato_vivo modelos``), como em ``test_ai.py``.
"""

import cv2
import numpy as np
import pytest

from retrato_vivo import damage, damage_ai, inpaint, metrics, models, runtime
from retrato_vivo.ai import ddcolor, face_restore
from retrato_vivo.aging import AgingConfig, age_photo
from retrato_vivo.pipeline import RestorationConfig, restore

from test_restauracao import make_scene, sepia_gradient

needs_runtime = pytest.mark.skipif(not runtime.is_available(), reason="onnxruntime não instalado")
needs_lama = pytest.mark.skipif(not inpaint.lama_available(), reason="modelo LaMa não baixado")
needs_detector = pytest.mark.skipif(not damage_ai.is_available(), reason="detector de riscos não baixado")
needs_gfpgan = pytest.mark.skipif(not face_restore.is_available(), reason="GFPGAN não baixado")
needs_ddcolor = pytest.mark.skipif(not ddcolor.is_available(), reason="DDColor não baixado")


def hole_scene():
    """Cena com um buraco largo preenchido de preto (como um rasgo marcado com o pincel)."""
    clean = make_scene()
    mask = np.zeros(clean.shape[:2], np.uint8)
    cv2.ellipse(mask, (400, 300), (60, 90), 20, 0, 360, 255, -1)
    holed = clean.copy()
    holed[mask > 0] = 0
    return clean, holed, mask


@pytest.fixture
def force_cpu(monkeypatch):
    """Simula um PC sem placa NVIDIA."""
    monkeypatch.setenv("RETRATO_VIVO_CPU", "1")
    runtime._gpu_usable.cache_clear()
    runtime.session.cache_clear()
    yield
    monkeypatch.delenv("RETRATO_VIVO_CPU")
    runtime._gpu_usable.cache_clear()
    runtime.session.cache_clear()


# ================================================================ runtime.py

@needs_runtime
def test_cpu_can_be_forced(force_cpu):
    assert runtime.device_name() == "CPU"


@needs_lama
def test_lama_gives_same_result_on_cpu_and_gpu(force_cpu):
    """Qualquer PC: a CPU dá o mesmo resultado da GPU (só demora mais)."""
    _, holed, mask = hole_scene()
    on_cpu = inpaint.inpaint(holed, mask, "ia")
    assert runtime.session_device(runtime.session(models.model_path("lama"))) == "CPU"
    runtime._gpu_usable.cache_clear()
    runtime.session.cache_clear()
    import os
    os.environ.pop("RETRATO_VIVO_CPU")
    on_default = inpaint.inpaint(holed, mask, "ia")
    os.environ["RETRATO_VIVO_CPU"] = "1"      # o fixture remove no fim
    assert np.abs(on_cpu.astype(int) - on_default.astype(int)).mean() < 2.0


# ================================================================ inpaint.py

def test_ia_falls_back_to_telea_without_the_model(monkeypatch):
    monkeypatch.setattr(inpaint, "lama_available", lambda: False)
    _, holed, mask = hole_scene()
    assert np.array_equal(inpaint.inpaint(holed, mask, "ia"), inpaint.inpaint(holed, mask, "telea"))
    assert "não baixado" in inpaint.last_note


@needs_lama
def test_lama_fills_large_hole_better_than_telea():
    clean, holed, mask = hole_scene()
    hole = mask > 0

    def error(img):
        return np.abs(img[hole].astype(int) - clean[hole].astype(int)).mean()

    with_ai = inpaint.inpaint(holed, mask, "ia")
    assert "LaMa" in inpaint.last_note
    assert error(with_ai) < error(inpaint.inpaint(holed, mask, "telea"))
    outside = ~cv2.dilate(mask, np.ones((15, 15), np.uint8)).astype(bool)    # além da borda suave
    assert np.array_equal(with_ai[outside], holed[outside])                    # fora do buraco nada muda


@needs_lama
def test_thin_scratches_stay_with_telea():
    damaged, _ = age_photo(make_scene(), AgingConfig.only_damage(seed=5))
    mask = np.zeros(damaged.shape[:2], np.uint8)
    cv2.line(mask, (50, 50), (700, 400), 255, 3)
    assert np.array_equal(inpaint.inpaint(damaged, mask, "ia"), inpaint.inpaint(damaged, mask, "telea"))
    assert "Telea" in inpaint.last_note


def test_pipeline_uses_ai_fill_by_default():
    assert RestorationConfig().inpaint_method == "ia"
    res = restore(make_scene())
    assert res.notes["danos_reparados"]                      # explica o que o preenchimento fez


# ============================================================== damage_ai.py

@needs_detector
def test_ai_detector_finds_synthetic_scratches():
    clean = make_scene()
    damaged, truth = age_photo(clean, AgingConfig(seed=3, n_spots=0, sepia=0, fade=0, grain=0,
                                                  vignette=0, blur=0))
    found, report = damage_ai.detect_scratches_ai(damaged)
    scores = metrics.mask_scores(found.astype(np.uint8), truth, tolerance_px=4)
    assert scores["revocacao"] > 0.5, scores
    assert report["aceitos"] >= 1


@needs_detector
def test_ai_detector_is_quiet_on_clean_photo():
    found, _ = damage_ai.detect_scratches_ai(make_scene())
    assert found.mean() < 0.01


@needs_detector
def test_detect_damage_adds_ai_scratches_only_when_enabled():
    damaged, _ = age_photo(make_scene(), AgingConfig(seed=3))
    on = damage.detect_damage(damaged)
    off = damage.detect_damage(damaged, damage.DamageConfig(use_ai=False))
    assert on.ai_scratches is not None and off.ai_scratches is None
    assert (on.mask > 0).sum() >= (off.mask > 0).sum()
    assert "riscos_ia" in on.report and on.stats["riscos_ia"] >= 1


def test_ai_threshold_follows_sensitivity():
    assert damage_ai.threshold_for(0.0) > damage_ai.threshold_for(0.5) > damage_ai.threshold_for(1.0)
    assert abs(damage_ai.threshold_for(0.5) - 0.4) < 0.05      # perto do valor dos autores


# ========================================================== face_restore.py

@needs_gfpgan
def test_face_stage_without_faces_changes_nothing():
    img = make_scene()
    stage = face_restore.make_face_restore_stage()
    res = restore(img, post_stages=[("rostos_ia", stage)])
    assert "nenhum rosto" in res.notes["rostos_ia"]
    assert np.array_equal(res.steps["rostos_ia"], res.steps[list(res.steps)[-2]])


def test_face_stage_explains_missing_model(monkeypatch):
    monkeypatch.setattr(face_restore, "is_available", lambda: False)
    res = restore(make_scene(), post_stages=[("rostos_ia", face_restore.make_face_restore_stage())])
    assert "não baixado" in res.notes["rostos_ia"]


# =============================================================== ddcolor.py

@needs_ddcolor
def test_ddcolor_colors_a_sepia_photo():
    res = restore(sepia_gradient(), post_stages=[("colorizacao", ddcolor.make_ddcolor_stage())])
    out = res.image.astype(int)
    assert "DDColor" in res.notes["colorizacao"]
    assert np.abs(out[..., 2] - out[..., 0]).max() > 5          # ganhou cor
    # A luminosidade é a da foto: só a cor muda.
    gray_in = cv2.cvtColor(res.steps["nitidez"], cv2.COLOR_BGR2GRAY).astype(int)
    gray_out = cv2.cvtColor(res.image, cv2.COLOR_BGR2GRAY).astype(int)
    assert np.abs(gray_in - gray_out).mean() < 12


@needs_ddcolor
def test_ddcolor_leaves_color_photo_alone():
    res = restore(make_scene(), post_stages=[("colorizacao", ddcolor.make_ddcolor_stage())])
    assert res.notes["colorizacao"].startswith("pulada")
