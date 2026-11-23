"""O pipeline: chama cada etapa na ordem certa. É o ponto de integração do projeto.

Ordem das etapas (a chave entre parênteses é o nome em ``result.steps``)::

    1. abrir e redimensionar            (original)
    1b. corrigir a perspectiva, opcional (papel_detectado, perspectiva)  perspective.py
    2. diagnosticar a foto              -> result.diagnosis
    3. tirar a dominante de cor         (cor_neutralizada)        preprocess.py
    4. recuperar o contraste            (niveis_automaticos)      preprocess.py
    5. detectar riscos, vincos, manchas (mapa_de_danos)           damage.py + faces.py
    6. apagar os danos                  (danos_reparados)         inpaint.py
    7. tirar o ruído                    (ruido_removido)          preprocess.py
    8. realçar contraste local          (clahe)                   preprocess.py
    9. nitidez                          (nitidez)                 preprocess.py
    10. etapas de IA plugadas           (nome dado por quem pluga)

Etapas que a foto não precisa são puladas; o motivo fica em ``result.notes``.

Contrato para plugar etapas novas:

- Imagens são sempre ``np.ndarray`` BGR ``uint8`` (H, W, 3).
- *Post stage*: ``funcao(img, result) -> img``. Pode consultar
  ``result.diagnosis.monochrome`` (só colorir P&B), ``result.damage.faces`` etc.
- *Inpainter* alternativo: ``funcao(img, mask_uint8) -> img``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import damage as dmg
from . import preprocess as pre
from . import inpaint as inpaint_module
from . import perspective
from .inpaint import inpaint
from .io_utils import resize_max_side
from .visualize import damage_map

# Texto de cada etapa, para a GUI mostrar ao usuário.
STEP_INFO: dict[str, tuple[str, str]] = {
    "original": ("Original", "A foto como foi aberta."),
    "papel_detectado": ("Papel encontrado", "Em verde, as bordas da foto impressa achadas na foto do celular."),
    "perspectiva": ("Perspectiva corrigida", "A foto impressa desentortada e recortada, como num scanner."),
    "cor_neutralizada": ("Cor neutralizada", "Remove o amarelado ou o tom sépia."),
    "niveis_automaticos": ("Contraste recuperado", "Devolve o preto e o branco a uma foto desbotada."),
    "mapa_de_danos": ("Mapa de danos", "Azul: riscos achados pela IA. Vermelho: riscos. Laranja: vincos. "
                                       "Amarelo: manchas. Verde: rostos protegidos."),
    "danos_reparados": ("Danos reparados", "Riscos finos preenchidos a partir da vizinhança; "
                                           "rasgos e áreas grandes, pela IA (LaMa)."),
    "ruido_removido": ("Ruído removido", "Reduz a granulação preservando as bordas."),
    "clahe": ("Contraste local", "Recupera detalhes em áreas muito escuras ou muito claras."),
    "nitidez": ("Nitidez", "Realça os contornos."),
    # etapas de IA (retrato_vivo/ai)
    "colorizacao": ("Colorização", "Uma rede neural prevê as cores de uma foto preto e branco."),
    "rostos_ia": ("Rostos restaurados", "Uma rede neural (GFPGAN) reconstrói olhos, boca e pele dos rostos."),
    "super_resolucao": ("Super-resolução", "Uma rede neural amplia a foto recriando parte do detalhe."),
}

Inpainter = Callable[[np.ndarray, np.ndarray], np.ndarray]

MIN_MARKED_AREA = 0.01      # fração da foto: cantos marcados à mão que cercam menos que isso não são o papel


@dataclass
class RestorationConfig:
    max_side: int | None = 2048            # fotos maiores são reduzidas para este tamanho
    correct_perspective: bool = False      # "foto de uma foto" tirada com o celular (perspective.py)
    # Os 4 cantos do papel marcados à mão: (x, y) em pixels da foto como foi aberta, antes de
    # qualquer redução. ``None``: o pipeline procura o papel sozinho.
    perspective_corners: np.ndarray | None = None
    preprocess: pre.PreprocessConfig = field(default_factory=pre.PreprocessConfig)
    damage: dmg.DamageConfig = field(default_factory=dmg.DamageConfig)
    repair_damage: bool = True
    inpaint_method: str = "ia"             # "ia" (LaMa nos buracos largos), "telea" ou "ns" (ver inpaint.py)
    inpaint_radius: float = 3.0


@dataclass
class RestorationResult:
    image: np.ndarray                                  # resultado final
    original: np.ndarray                               # entrada já redimensionada
    scale: float                                       # redução aplicada na entrada (1 = nenhuma)
    diagnosis: pre.Diagnosis                           # o que foi medido na foto
    damage: dmg.DamageResult | None = None             # máscara, estatísticas e relatório
    steps: dict[str, np.ndarray] = field(default_factory=dict)   # imagem de cada etapa aplicada
    notes: dict[str, str] = field(default_factory=dict)          # por que cada etapa agiu ou foi pulada
    timings_ms: dict[str, float] = field(default_factory=dict)   # tempo de cada fase
    # Os 4 cantos do papel usados na correção de perspectiva, em pixels da foto como foi aberta.
    # ``None``: a correção estava desligada ou não havia papel para desentortar.
    paper_corners: np.ndarray | None = None


PostStage = Callable[[np.ndarray, RestorationResult], np.ndarray]


class _Timer:
    """Mede o tempo de um bloco ``with`` e guarda em milissegundos."""

    def __init__(self, sink: dict[str, float], name: str):
        self.sink, self.name = sink, name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        self.sink[self.name] = (time.perf_counter() - self.t0) * 1000.0


def _marked_corners(corners: np.ndarray, scale: float, shape: tuple[int, ...]) -> np.ndarray | None:
    """Os cantos marcados à mão, na escala da foto reduzida e em ordem.

    ``None`` se eles não cercam uma área que possa ser o papel: cantos repetidos ou
    alinhados (a transformação de perspectiva não existe) ou uma área minúscula.
    """
    quad = perspective.ordenar_cantos(np.asarray(corners, np.float32) * scale)
    side = np.roll(quad, -1, axis=0) - quad                 # lado i: do canto i ao canto seguinte
    next_side = np.roll(side, -1, axis=0)
    # Produto vetorial de cada lado com o seguinte: positivo nos 4 cantos quando o quadrilátero é convexo.
    turn = side[:, 0] * next_side[:, 1] - side[:, 1] * next_side[:, 0]
    if (turn <= 0).any() or cv2.contourArea(quad) < MIN_MARKED_AREA * shape[0] * shape[1]:
        return None
    return quad


def restore(image: np.ndarray,
            config: RestorationConfig | None = None,
            mask_add: np.ndarray | None = None,
            mask_remove: np.ndarray | None = None,
            inpainter: Inpainter | None = None,
            post_stages: list[tuple[str, PostStage]] | None = None) -> RestorationResult:
    """Restaura uma foto.

    ``mask_add``/``mask_remove`` vêm do editor de máscara da GUI e devem ter
    o tamanho de ``result.original`` (a foto já redimensionada). Fluxo
    típico: chamar ``restore``, mostrar o mapa de danos, o usuário corrige a
    máscara e ``restore`` é chamado de novo com as correções.
    """
    cfg = config or RestorationConfig()
    timings: dict[str, float] = {}
    log = pre.StepLog()

    with _Timer(timings, "abrir"):
        img, scale = resize_max_side(image, cfg.max_side)
    log.add("original", img)

    corners = None
    if cfg.correct_perspective:
        # Antes de tudo: o resto do pipeline (e o "antes" da comparação) já vê a foto desentortada.
        with _Timer(timings, "perspectiva"):
            by_hand = cfg.perspective_corners is not None
            if by_hand:
                corners = _marked_corners(cfg.perspective_corners, scale, img.shape)
            else:
                corners = perspective.encontrar_papel(img)
            if corners is None:
                log.skip("perspectiva", ("os cantos marcados não cercam uma área da foto" if by_hand else
                                         "não achei as bordas de uma foto impressa") + "; a foto segue como está")
            else:
                log.add("papel_detectado", perspective.desenhar_cantos(img, corners))
                img = perspective.retificar(img, corners)
                log.add("perspectiva", img, f"papel desentortado e recortado: {img.shape[1]}x{img.shape[0]} px"
                        + (", com os cantos marcados à mão" if by_hand else ""))

    with _Timer(timings, "diagnostico"):
        diagnosis = pre.diagnose(img)

    with _Timer(timings, "normalizar"):
        work = pre.normalize(img, cfg.preprocess, diagnosis, log)

    result = RestorationResult(image=work, original=img, scale=scale, diagnosis=diagnosis,
                               steps=log.steps, notes=log.notes, timings_ms=timings,
                               paper_corners=None if corners is None else corners / scale)

    if cfg.repair_damage:
        with _Timer(timings, "detectar_danos"):
            result.damage = dmg.detect_damage(work, cfg.damage, mask_add, mask_remove)
        stats = result.damage.stats
        log.add("mapa_de_danos", damage_map(work, result.damage),
                f"{stats['riscos']:.0f} riscos, {stats['vincos']:.0f} vincos, {stats['manchas']:.0f} manchas"
                + (f", {stats['riscos_ia']:.0f} riscos pela IA" if result.damage.ai_scratches is not None else
                   " (detector de riscos com IA não baixado)")
                + f", {stats['area_danificada_pct']:.2f}% da área")
        with _Timer(timings, "inpainting"):
            if inpainter is not None:
                work = inpainter(work, result.damage.mask)
            else:
                work = inpaint(work, result.damage.mask, cfg.inpaint_method, cfg.inpaint_radius)
        method_note = {"telea": "Telea (clássico)", "ns": "Navier-Stokes (clássico)"}.get(
            cfg.inpaint_method, inpaint_module.last_note)
        log.add("danos_reparados", work, "personalizado" if inpainter is not None else method_note)

    with _Timer(timings, "realce"):
        work = pre.enhance(work, cfg.preprocess, log)

    for name, stage in post_stages or []:
        with _Timer(timings, name):
            work = stage(work, result)
        log.add(name, work)

    result.image = work
    return result
