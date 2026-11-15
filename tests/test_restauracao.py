"""Testes da parte de pré-processamento e danos. Rodar com ``python -m pytest``.

Organização: um bloco por arquivo do pacote, na ordem do pipeline.
"""

import cv2
import numpy as np
import pytest

from retrato_vivo import damage, metrics, preprocess
from retrato_vivo import aging
from retrato_vivo.aging import AgingConfig, age_photo, grain, sepia
from retrato_vivo.cli import explain
from retrato_vivo.inpaint import inpaint
from retrato_vivo.io_utils import load_image, resize_max_side, save_image, to_bgr8
from retrato_vivo.pipeline import STEP_INFO, RestorationConfig, restore
from retrato_vivo.visualize import side_by_side, steps_panel


def make_scene(h: int = 600, w: int = 800, seed: int = 0) -> np.ndarray:
    """Cena sintética "fotográfica": gradiente, formas de borda suave e textura leve.
    Tem regiões neutras (cinzas), como qualquer foto real sem dominante de cor."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:h, :w].astype(np.float32)
    base = 60 + 120 * (x / w) + 40 * np.sin(y / h * np.pi)
    img = np.dstack([base, base, base])
    for _ in range(6):
        c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        r = int(rng.uniform(40, 120))
        color = tuple(float(v) for v in rng.uniform(30, 230, 3))
        cv2.circle(img, c, r, color, -1)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    img += cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 1.5)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def damaged_scene(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = make_scene()
    damaged, truth = age_photo(clean, AgingConfig.only_damage(seed=seed))
    return clean, damaged, truth


# ============================================================== io_utils.py

def test_to_bgr8_variants():
    gray = np.full((10, 12), 100, np.uint8)
    assert to_bgr8(gray).shape == (10, 12, 3)
    rgba = np.zeros((10, 12, 4), np.uint8)  # totalmente transparente -> branco
    assert (to_bgr8(rgba) == 255).all()
    deep = np.full((10, 12, 3), 65535, np.uint16)
    assert (to_bgr8(deep) == 255).all()


def test_roundtrip_unicode_path(tmp_path):
    img = make_scene(50, 70)
    p = save_image(tmp_path / "fotos_da_vovó" / "imagem_ção.png", img)
    assert np.array_equal(load_image(p), img)


def test_resize_never_upscales():
    img = make_scene(100, 200)
    assert resize_max_side(img, 1000)[0].shape == img.shape
    small, s = resize_max_side(img, 100)
    assert small.shape[:2] == (50, 100) and s == pytest.approx(0.5)


def test_unreadable_file_raises_value_error(tmp_path):
    fake = tmp_path / "foto.jpg"  # ex.: AVIF sem suporte ou lixo com extensão .jpg
    fake.write_bytes(b"\x00\x00\x00 ftypavif" + b"\x00" * 64)
    with pytest.raises(ValueError):
        load_image(fake)


def test_exif_rotation_is_applied(tmp_path):
    """Foto de celular tirada em pé: o arquivo é deitado + etiqueta EXIF de rotação."""
    Image = pytest.importorskip("PIL.Image")
    landscape = Image.fromarray(np.zeros((40, 80, 3), np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6  # "girar 90 graus para exibir"
    path = tmp_path / "celular.jpg"
    landscape.save(path, exif=exif)
    assert load_image(path).shape[:2] == (80, 40)


# ============================================================ preprocess.py

def test_sepia_is_monochrome_and_becomes_neutral_gray():
    sepia_photo = sepia(preprocess.to_neutral_gray(make_scene()), 1.0)
    diag = preprocess.diagnose(sepia_photo)
    assert diag.monochrome and diag.cast_strength > 5
    log = preprocess.StepLog()
    out = preprocess.normalize(sepia_photo, preprocess.PreprocessConfig(), diag, log)
    assert preprocess.is_gray(out) and "cor_neutralizada" in log.steps


def test_good_color_photo_keeps_its_colors():
    """Foto colorida sem dominante: a etapa de cor é pulada e explicada."""
    img = make_scene()
    diag = preprocess.diagnose(img)
    assert not diag.monochrome and diag.cast_strength < preprocess.MIN_CAST_TO_CORRECT
    log = preprocess.StepLog()
    cfg = preprocess.PreprocessConfig(auto_levels=False)
    assert np.array_equal(preprocess.normalize(img, cfg, diag, log), img)
    assert log.notes["cor_neutralizada"].startswith("pulada")


def test_yellowed_color_photo_gets_corrected():
    clean = make_scene()
    yellowed = sepia(clean, 0.35)
    diag = preprocess.diagnose(yellowed)
    assert not diag.monochrome and diag.cast_strength >= preprocess.MIN_CAST_TO_CORRECT
    fixed = preprocess.remove_color_cast(yellowed, diag.cast_ab, diag.cast_ref_l)

    def mean_b(img):  # eixo azul-amarelo do LAB
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 2].mean()

    assert abs(mean_b(fixed) - mean_b(clean)) < abs(mean_b(yellowed) - mean_b(clean))


def test_auto_levels_expands_faded_photo():
    faded = (make_scene() * 0.4 + 100).astype(np.uint8)
    out, gain = preprocess.auto_levels(faded)
    assert gain > 1.2 and out.std() > faded.std() * 1.2


def test_auto_levels_leaves_full_range_photo_alone():
    img = make_scene()
    img[:20], img[-20:] = 0, 255
    out, gain = preprocess.auto_levels(img)
    assert gain == 1.0 and out is img


def test_auto_levels_does_not_blow_out_highlights():
    img = make_scene()
    img[:150] = 175  # retrato de estúdio: o fundo claro é o topo do histograma
    out, _ = preprocess.auto_levels(img)
    assert out[:150].mean() < 245


def test_noise_estimate_and_dose():
    clean = cv2.GaussianBlur(make_scene(), (0, 0), 2)
    noisy = grain(clean, 8.0, np.random.default_rng(0))
    assert preprocess.estimate_noise(clean) < preprocess.NOISE_TO_DENOISE
    assert preprocess.estimate_noise(noisy) == pytest.approx(8.0, abs=2.0)
    assert preprocess.denoise_dose(0.7) == 0.0          # foto limpa: não mexe
    assert preprocess.denoise_dose(8.0) > preprocess.denoise_dose(4.0) > 0


def test_enhance_skips_denoise_on_clean_photo():
    clean = cv2.GaussianBlur(make_scene(), (0, 0), 2)
    log = preprocess.StepLog()
    preprocess.enhance(clean, preprocess.PreprocessConfig(), log)
    assert "ruido_removido" not in log.steps
    assert log.notes["ruido_removido"].startswith("pulada")


# ================================================================ damage.py

def test_clean_image_has_few_false_positives():
    res = damage.detect_damage(make_scene())
    assert res.stats["area_danificada_pct"] < 0.5


def test_detects_synthetic_damage():
    _, damaged, truth = damaged_scene(seed=3)
    res = damage.detect_damage(damaged)
    s = metrics.mask_scores(res.mask, truth)
    assert s["revocacao"] > 0.6, s
    assert s["precisao"] > 0.6, s


def test_detects_wide_white_crease_but_not_on_clean_photo():
    """Vinco: linha branca mais larga que um risco (dobra ou rachadura do papel)."""
    clean = make_scene()
    assert damage.detect_damage(clean).stats["vincos"] == 0
    creased = clean.copy()
    cv2.line(creased, (100, 80), (650, 520), (250, 250, 250), 6, cv2.LINE_AA)
    res = damage.detect_damage(creased)
    truth = cv2.absdiff(creased, clean).max(axis=2) > 12
    assert res.stats["vincos"] >= 1
    assert (res.creases & truth).sum() > 0.6 * truth.sum()


def test_report_counts_every_candidate():
    """O relatório fecha a conta: candidatos = descartados + aceitos."""
    _, damaged, _ = damaged_scene(seed=3)
    res = damage.detect_damage(damaged)
    for detector in ("riscos", "vincos", "manchas"):
        report = dict(res.report[detector])
        candidates, accepted = report.pop("candidatos"), report.pop("aceitos")
        assert candidates == accepted + sum(report.values())
        assert accepted == res.stats[detector]


def test_user_mask_add_and_remove():
    img = make_scene(200, 200)
    add = np.zeros(img.shape[:2], np.uint8)
    add[10:20, 10:20] = 255
    res = damage.detect_damage(img, mask_add=add)
    assert (res.mask[10:20, 10:20] == 255).all()
    res = damage.detect_damage(img, mask_add=add, mask_remove=add)
    assert not res.mask[10:20, 10:20].any()


# =============================================================== inpaint.py

@pytest.mark.parametrize("method", ["telea", "ns"])
def test_inpainting_improves_quality(method):
    clean, damaged, _ = damaged_scene(seed=5)
    res = damage.detect_damage(damaged)
    repaired = inpaint(damaged, res.mask, method)
    assert metrics.psnr(clean, repaired) > metrics.psnr(clean, damaged) + 3
    assert metrics.ssim(clean, repaired) > metrics.ssim(clean, damaged)


def test_inpaint_rejects_unknown_method():
    img = make_scene(50, 50)
    with pytest.raises(ValueError):
        inpaint(img, np.zeros(img.shape[:2], np.uint8), "magica")


# ============================================================== pipeline.py

@pytest.mark.parametrize("shape", [(37, 53, 3), (480, 640, 3), (900, 300, 3), (300, 300)])
def test_pipeline_any_size(shape):
    img = to_bgr8(np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8))
    res = restore(img)
    assert res.image.shape == res.original.shape and res.image.dtype == np.uint8
    assert "mapa_de_danos" in res.steps


def test_pipeline_does_little_to_a_good_photo():
    """Regra de ouro: foto boa entra, foto quase igual sai."""
    good = cv2.GaussianBlur(make_scene(), (0, 0), 1.5)
    good[:20], good[-20:] = 5, 250          # já usa toda a faixa de tons
    res = restore(good)
    assert metrics.psnr(good, res.image) > 28
    assert res.notes["ruido_removido"].startswith("pulada")
    assert res.notes["niveis_automaticos"].startswith("pulada")


def test_every_builtin_step_has_a_description():
    damaged, _ = age_photo(make_scene(), AgingConfig(seed=1))
    res = restore(damaged)
    assert set(res.steps) <= set(STEP_INFO)
    assert "diagnóstico" in explain(res)


def test_pipeline_post_stage_and_custom_inpainter():
    calls = []

    def fake_inpainter(img, mask):
        calls.append("inpaint")
        return img

    def invert(img, result):
        calls.append(("post", result.diagnosis.monochrome))
        return 255 - img

    res = restore(make_scene(), RestorationConfig(), inpainter=fake_inpainter,
                  post_stages=[("inverter", invert)])
    assert calls[0] == "inpaint" and calls[1][0] == "post"
    assert "inverter" in res.steps and "inverter" in res.timings_ms


# ============================================================= visualize.py

def test_visualizations_have_sane_shapes():
    res = restore(make_scene(300, 400))
    assert side_by_side(res.original, res.image).shape[0] == 300
    panel = steps_panel(res.steps, columns=3, cell_height=100)
    assert panel.ndim == 3 and panel.shape[1] > 400


# ================================================================= aging.py
# Efeitos usados pela Máquina do tempo.

def test_tone_effects_keep_size_and_type():
    img = make_scene(120, 160)
    rng = np.random.default_rng(0)
    effects = [aging.black_and_white, aging.sepia, aging.fade, aging.contrast,
               aging.saturation, lambda x: aging.color_cast(x, (0.8, 1.0, 1.1), (5, 10, 20)),
               aging.soften, aging.vignette, lambda x: aging.grain(x, 5, rng),
               lambda x: aging.add_scratches(x, 5, rng), lambda x: aging.add_spots(x, 20, rng),
               lambda x: aging.date_stamp(x, "12 08 '95")]
    for fx in effects:
        out = fx(img)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_black_and_white_and_saturation_zero_are_gray():
    img = make_scene()
    assert preprocess.is_gray(aging.black_and_white(img))
    assert preprocess.diagnose(aging.saturation(img, 0.0)).monochrome


def test_contrast_and_color_cast_do_what_they_say():
    img = make_scene()
    assert aging.contrast(img, 1.5).std() > img.std() > aging.contrast(img, 0.6).std()
    warm = aging.color_cast(img, (0.8, 1.0, 1.1))
    assert warm[:, :, 2].mean() > img[:, :, 2].mean()   # mais vermelho
    assert warm[:, :, 0].mean() < img[:, :, 0].mean()   # menos azul


def test_frames_grow_the_image():
    img = make_scene(300, 400)
    assert aging.crop_square(img).shape[:2] == (300, 300)
    bordered = aging.add_border(img, 0.05)
    assert bordered.shape[0] == 300 + 2 * 20 and bordered.shape[1] == 400 + 2 * 20
    polaroid = aging.polaroid_frame(aging.crop_square(img))
    assert polaroid.shape[0] > polaroid.shape[1]          # base mais larga que o topo


def test_date_stamp_draws_orange_in_the_corner():
    img = np.zeros((400, 400, 3), np.uint8)
    out = aging.date_stamp(img, "12 08 '95")
    corner = out[300:, 200:]
    orange = (corner[:, :, 2] > 200) & (corner[:, :, 0] < 80)
    assert orange.sum() > 50 and not out[:100, :100].any()


def test_aged_photo_is_restorable():
    """Ida e volta: foto envelhecida com os efeitos da Máquina do tempo é
    reconhecida como P&B/sépia e tem os danos detectados pela restauração."""
    rng = np.random.default_rng(3)
    old = aging.add_scratches(aging.grain(aging.sepia(aging.black_and_white(make_scene()), 0.6), 6, rng), 10, rng)
    res = restore(old)
    assert res.diagnosis.monochrome
    assert res.damage.stats["riscos"] + res.damage.stats["vincos"] >= 3
