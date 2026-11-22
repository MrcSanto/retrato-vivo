"""Testes das etapas de Deep Learning (retrato_vivo/ai).

Os testes que rodam a rede são pulados se o modelo não foi baixado
(``python -m retrato_vivo modelos``); os demais rodam sempre.
"""

import importlib

import cv2
import numpy as np
import pytest

from retrato_vivo import preprocess
from retrato_vivo.ai import colorize, colorize_image, make_colorize_stage
from retrato_vivo.cli import build_parser
from retrato_vivo.pipeline import STEP_INFO, restore

# ``retrato_vivo.ai.colorize`` é a função (exportada no __init__); o módulo vem daqui.
colorize_module = importlib.import_module("retrato_vivo.ai.colorize")

needs_colorization_model = pytest.mark.skipif(
    not colorize_module.is_available(),
    reason="modelo de colorização não baixado (python -m retrato_vivo modelos)")


def make_color_scene(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """Cena colorida simples: céu azul, grama verde e círculos de cores variadas."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = (200, 150, 90)       # céu (BGR)
    img[h // 2:] = (60, 140, 50)         # grama
    for _ in range(5):
        c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        color = tuple(int(v) for v in rng.uniform(30, 230, 3))
        cv2.circle(img, c, int(rng.uniform(15, 50)), color, -1)
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def gray_scene() -> np.ndarray:
    return preprocess.to_neutral_gray(make_color_scene())


def chroma(img: np.ndarray) -> float:
    """Saturação média no LAB (0 = cinza)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128).mean())


def lightness(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)


# ============================================================ colorize.py

@needs_colorization_model
def test_colorize_adds_color_and_keeps_the_details():
    gray = gray_scene()
    out = colorize_image(gray, strength=1.0)
    assert out.shape == gray.shape and out.dtype == np.uint8
    assert chroma(gray) < 1.0 and chroma(out) > 3.0
    # a luminosidade (os detalhes da foto) é a da original
    assert np.abs(lightness(out) - lightness(gray)).mean() < 2.0


@needs_colorization_model
def test_colorize_strength_controls_how_much_color():
    gray = gray_scene()
    none, half, full = (colorize_image(gray, s) for s in (0.0, 0.5, 1.0))
    assert chroma(none) < 1.0
    assert chroma(none) < chroma(half) < chroma(full)


@needs_colorization_model
@pytest.mark.parametrize("shape", [(37, 501), (480, 64), (300, 300)])
def test_colorize_any_size(shape):
    gray = cv2.resize(gray_scene(), shape[::-1])
    assert colorize_image(gray).shape == gray.shape


@needs_colorization_model
def test_pipeline_colorizes_black_and_white_photo():
    res = restore(gray_scene(), post_stages=[("colorizacao", colorize)])
    assert res.diagnosis.monochrome
    assert "colorizacao" in res.steps and "colorizacao" in res.timings_ms
    assert res.notes["colorizacao"].startswith("foto P&B")
    assert chroma(res.image) > 3.0


def test_color_photo_passes_through_untouched():
    """Não precisa do modelo: a etapa decide antes de carregar a rede."""
    photo = make_color_scene()
    res = restore(photo, post_stages=[("colorizacao", colorize)])
    assert not res.diagnosis.monochrome
    assert res.notes["colorizacao"].startswith("pulada")
    assert np.array_equal(res.steps["colorizacao"], res.steps.get("nitidez", res.image))


def test_missing_model_is_skipped_with_explanation(monkeypatch):
    monkeypatch.setattr(colorize_module, "is_available", lambda: False)
    gray = gray_scene()
    res = restore(gray, post_stages=[("colorizacao", make_colorize_stage())])
    assert "modelo não baixado" in res.notes["colorizacao"]
    assert chroma(res.image) < 1.0


def test_colorization_step_has_a_description():
    assert "colorizacao" in STEP_INFO


def test_cli_accepts_colorization_options():
    args = build_parser().parse_args(["restaurar", "foto.jpg", "--colorir", "--intensidade-cor", "0.6"])
    assert args.colorir and args.intensidade_cor == 0.6


# ============================================================ superres.py

superres = importlib.import_module("retrato_vivo.ai.superres")
tf_fix = importlib.import_module("retrato_vivo.ai.tf_fix")

needs_espcn = pytest.mark.skipif(
    not (superres.is_available("espcn", 2) and superres.is_available("espcn", 4)),
    reason="modelos ESPCN não baixados (python -m retrato_vivo modelos)")
needs_edsr = pytest.mark.skipif(
    not superres.is_available("edsr", 2),
    reason="modelo EDSR não baixado (python -m retrato_vivo modelos)")


def test_tf_fix_adds_blocksize_to_depth_to_space():
    """Grafo mínimo montado à mão: um nó comum e um DepthToSpace com block_size=3."""
    f = tf_fix._message_field
    attr_value_i3 = tf_fix._varint(3 << 3 | 0) + tf_fix._varint(3)        # AttrValue { i: 3 }
    attr = f(5, f(1, b"block_size") + f(2, attr_value_i3))
    graph = f(1, f(1, b"conv") + f(2, b"Conv2D")) + f(1, f(1, b"d2s") + f(2, b"DepthToSpace") + attr)

    fixed_graph, n = tf_fix.fix_depth_to_space(graph)
    assert n == 1 and b"blocksize" in fixed_graph and len(fixed_graph) > len(graph)
    assert tf_fix.fix_depth_to_space(fixed_graph)[1] == 0      # aplicar de novo não muda nada


@pytest.mark.parametrize("side,target,expected", [
    (2048, 2048, None), (1700, 2048, None),   # já grande (ou quase)
    (1200, 2048, 2), (1024, 2048, 2),
    (800, 2048, 4), (300, 2048, 4),
])
def test_choose_factor(side, target, expected):
    assert superres.choose_factor(side, target) == expected


def test_invalid_model_or_factor_raises():
    with pytest.raises(ValueError):
        superres.make_upscale_stage("bicubico")
    with pytest.raises(ValueError):
        superres.model_name("espcn", 3)


def make_detail_scene(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """Cena com bordas finas e texto: o tipo de detalhe que a super-resolução recupera.
    (Em áreas lisas o bicúbico já é quase perfeito e não há o que ganhar.)"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 235, np.uint8)
    for _ in range(12):
        p1 = tuple(int(v) for v in rng.uniform(0, (w, h)))
        p2 = tuple(int(v) for v in rng.uniform(0, (w, h)))
        color = tuple(int(v) for v in rng.uniform(0, 200, 3))
        cv2.line(img, p1, p2, color, int(rng.uniform(1, 4)), cv2.LINE_AA)
    for y in range(30, h, 40):
        cv2.putText(img, "Retrato Vivo 1932", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (40, 30, 20), 2, cv2.LINE_AA)
    return img


@needs_espcn
@pytest.mark.parametrize("factor", [2, 4])
def test_espcn_upscales_exactly_and_beats_bicubic(factor):
    hr = make_detail_scene()
    lr = cv2.resize(hr, (320 // factor, 240 // factor), interpolation=cv2.INTER_CUBIC)
    out = superres.upscale_image(lr, factor, "espcn")
    assert out.shape == hr.shape and out.dtype == np.uint8
    bicubic = cv2.resize(lr, (320, 240), interpolation=cv2.INTER_CUBIC)
    assert cv2.PSNR(hr, out) > cv2.PSNR(hr, bicubic)


@needs_edsr
def test_edsr_upscales_small_image():
    lr = cv2.resize(make_color_scene(), (40, 30), interpolation=cv2.INTER_AREA)
    out = superres.upscale_image(lr, 2, "edsr")
    assert out.shape == (60, 80, 3) and out.dtype == np.uint8


def test_large_photo_is_not_upscaled():
    """Não precisa do modelo: a etapa decide pelo tamanho antes de carregar a rede."""
    res = restore(make_color_scene(1100, 1900), post_stages=[("super_resolucao", superres.upscale)])
    assert res.notes["super_resolucao"].startswith("pulada")
    assert res.image.shape[:2] == (1100, 1900)


@needs_espcn
def test_pipeline_upscales_small_photo_up_to_the_target():
    res = restore(make_color_scene(300, 400),
                  post_stages=[("super_resolucao", superres.make_upscale_stage("espcn", 1000))])
    assert max(res.image.shape[:2]) == 1000                 # 4x daria 1600: reduz até o alvo
    assert res.image.shape[1] / res.image.shape[0] == pytest.approx(400 / 300, rel=0.01)
    assert "ampliada 4x com ESPCN" in res.notes["super_resolucao"]
    assert "super_resolucao" in res.timings_ms and "super_resolucao" in STEP_INFO


@needs_espcn
def test_edsr_falls_back_to_espcn_on_big_photo():
    img = make_color_scene(600, 600)                        # 360 mil pixels > EDSR_MAX_PIXELS
    res = restore(img, post_stages=[("super_resolucao", superres.make_upscale_stage("edsr", 1200))])
    assert "ESPCN (EDSR seria lento demais" in res.notes["super_resolucao"]
    assert res.image.shape[:2] == (1200, 1200)


def test_missing_superres_model_is_skipped(monkeypatch):
    monkeypatch.setattr(superres, "is_available", lambda *a, **k: False)
    res = restore(make_color_scene(200, 300), post_stages=[("super_resolucao", superres.upscale)])
    assert "modelo não baixado" in res.notes["super_resolucao"]
    assert res.image.shape[:2] == (200, 300)


def test_cli_accepts_superres_options():
    args = build_parser().parse_args(["restaurar", "f.jpg", "--ampliar", "--modelo-sr", "edsr",
                                      "--tamanho-alvo", "1500"])
    assert args.ampliar and args.modelo_sr == "edsr" and args.tamanho_alvo == 1500


# ============================================================ edges.py

edges = importlib.import_module("retrato_vivo.ai.edges")

needs_hed = pytest.mark.skipif(
    not edges.is_available(),
    reason="modelo HED não baixado (python -m retrato_vivo modelos)")


def square_scene(h: int = 300, w: int = 400) -> np.ndarray:
    """Quadrado branco (x 100–300, y 80–220) sobre fundo preto."""
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (100, 80), (300, 220), (255, 255, 255), -1)
    return img


def test_canny_finds_the_square_outline():
    out = edges.bordas_canny(square_scene())
    assert out.shape == (300, 400) and out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}
    assert out[78:83, 150:250].any()            # borda de cima
    assert not out[120:180, 150:250].any()      # dentro do quadrado: liso, sem borda


def test_crop_layer_cuts_from_the_center():
    layer = edges._CropLayer(None, None)
    assert layer.getMemoryShapes([[1, 1, 10, 12], [1, 3, 6, 8]]) == [[1, 1, 6, 8]]
    blob = np.arange(120, dtype=np.float32).reshape(1, 1, 10, 12)
    out = layer.forward([blob, None])[0]
    assert out.shape == (1, 1, 6, 8)
    assert out[0, 0, 0, 0] == blob[0, 0, 2, 2]  # margem de (10-6)/2 = 2 e (12-8)/2 = 2


def test_panel_without_hed_still_comes_out(monkeypatch):
    monkeypatch.setattr(edges, "is_available", lambda: False)
    antes, depois = square_scene(300, 400), square_scene(600, 800)   # depois ampliada 2x
    panel = edges.painel_bordas(antes, depois, gap=8)
    cell_h = 300 + edges.LABEL_HEIGHT
    assert panel.shape == (2 * cell_h + 8, 400 + 8 + 400, 3)


@needs_hed
def test_hed_marks_the_square_outline_in_the_right_place():
    out = edges.bordas_hed(square_scene())
    assert out.shape == (300, 400) and out.dtype == np.uint8
    ys, xs = np.where(out > 128)
    # com o Crop do OpenCV 4.14 as bordas sairiam ~35 px deslocadas
    assert abs(xs.min() - 100) <= 5 and abs(xs.max() - 300) <= 5
    assert abs(ys.min() - 80) <= 5 and abs(ys.max() - 220) <= 5
    assert out[120:180, 150:250].max() < 30     # dentro do quadrado: liso


@needs_hed
def test_hed_handles_big_and_gray_photos():
    big = cv2.resize(square_scene(), (1200, 900), interpolation=cv2.INTER_NEAREST)
    assert edges.bordas_hed(big).shape == (900, 1200)
    assert edges.bordas_hed(gray_scene()).shape == gray_scene().shape[:2]


def test_hed_does_not_leave_the_crop_layer_registered(monkeypatch):
    """Outras redes Caffe (colorização) não podem ganhar a nossa Crop por tabela."""
    calls = []

    class FakeNet:
        def setInput(self, blob):
            calls.append("entrada")

        def forward(self):
            calls.append("executa")
            return np.zeros((1, 1, 30, 40), np.float32)

    monkeypatch.setattr(cv2, "dnn_registerLayer", lambda *a: calls.append("registra"))
    monkeypatch.setattr(cv2, "dnn_unregisterLayer", lambda *a: calls.append("remove"))
    monkeypatch.setattr(cv2.dnn, "readNetFromCaffe", lambda *a: FakeNet())
    monkeypatch.setattr(edges, "is_available", lambda: True)
    monkeypatch.setattr(edges, "_NET", None)
    edges.bordas_hed(np.zeros((30, 40, 3), np.uint8))
    # a Crop fica ligada no carregamento e na execução, e desligada no fim de cada um
    assert calls == ["registra", "remove", "entrada", "registra", "executa", "remove"]


def test_cli_accepts_edges_option():
    assert build_parser().parse_args(["restaurar", "f.jpg", "--bordas"]).bordas
