"""O pipeline: chama cada etapa na ordem certa. É o ponto de integração do projeto.

Ordem das etapas (a chave entre parênteses é o nome em ``result.steps``)::

    1. abrir e redimensionar            (original)
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

import numpy as np

from . import damage as dmg
from . import preprocess as pre
from .inpaint import inpaint
from .io_utils import resize_max_side
from .visualize import damage_map

# Texto de cada etapa, para a GUI mostrar ao usuário.
STEP_INFO: dict[str, tuple[str, str]] = {
    "original": ("Original", "A foto como foi aberta."),
    "cor_neutralizada": ("Cor neutralizada", "Remove o amarelado ou o tom sépia."),
    "niveis_automaticos": ("Contraste recuperado", "Devolve o preto e o branco a uma foto desbotada."),
    "mapa_de_danos": ("Mapa de danos", "Vermelho: riscos. Laranja: vincos. Amarelo: manchas. Verde: rostos protegidos."),
    "danos_reparados": ("Danos reparados", "Riscos e manchas preenchidos a partir da vizinhança."),
    "ruido_removido": ("Ruído removido", "Reduz a granulação preservando as bordas."),
    "clahe": ("Contraste local", "Recupera detalhes em áreas muito escuras ou muito claras."),
    "nitidez": ("Nitidez", "Realça os contornos."),
}

Inpainter = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class RestorationConfig:
    max_side: int | None = 2048            # fotos maiores são reduzidas para este tamanho
    preprocess: pre.PreprocessConfig = field(default_factory=pre.PreprocessConfig)
    damage: dmg.DamageConfig = field(default_factory=dmg.DamageConfig)
    repair_damage: bool = True
    inpaint_method: str = "telea"          # "telea" ou "ns" (ver inpaint.py)
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


PostStage = Callable[[np.ndarray, RestorationResult], np.ndarray]


class _Timer:
    """Mede o tempo de um bloco ``with`` e guarda em milissegundos."""

    def __init__(self, sink: dict[str, float], name: str):
        self.sink, self.name = sink, name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        self.sink[self.name] = (time.perf_counter() - self.t0) * 1000.0


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

    with _Timer(timings, "diagnostico"):
        diagnosis = pre.diagnose(img)

    with _Timer(timings, "normalizar"):
        work = pre.normalize(img, cfg.preprocess, diagnosis, log)

    result = RestorationResult(image=work, original=img, scale=scale, diagnosis=diagnosis,
                               steps=log.steps, notes=log.notes, timings_ms=timings)

    if cfg.repair_damage:
        with _Timer(timings, "detectar_danos"):
            result.damage = dmg.detect_damage(work, cfg.damage, mask_add, mask_remove)
        stats = result.damage.stats
        log.add("mapa_de_danos", damage_map(work, result.damage),
                f"{stats['riscos']:.0f} riscos, {stats['vincos']:.0f} vincos, {stats['manchas']:.0f} manchas, "
                f"{stats['area_danificada_pct']:.2f}% da área")
        with _Timer(timings, "inpainting"):
            if inpainter is not None:
                work = inpainter(work, result.damage.mask)
            else:
                work = inpaint(work, result.damage.mask, cfg.inpaint_method, cfg.inpaint_radius)
        log.add("danos_reparados", work)

    with _Timer(timings, "realce"):
        work = pre.enhance(work, cfg.preprocess, log)

    for name, stage in post_stages or []:
        with _Timer(timings, name):
            work = stage(work, result)
        log.add(name, work)

    result.image = work
    return result
