"""Detecção de danos: encontra riscos, vincos, manchas de poeira e rasgos na foto.

O resultado é uma MÁSCARA (imagem preto e branco em que branco = "aqui tem
dano"). Quem apaga o dano é ``inpaint.py``.

Como ler este arquivo
---------------------
Os detectores (riscos, vincos e manchas) seguem a mesma receita em 3 passos:

1. **Realçar**: uma operação que deixa em destaque tudo o que PARECE o dano
   (coisas finas para riscos; pontos diferentes da vizinhança para manchas).
2. **Agrupar**: os pixels realçados viram regiões candidatas (``Candidates``).
3. **Filtrar**: cada candidata passa por uma lista de testes com nome
   ("curto demais", "em borda da cena"...). Quem falha é descartada, e o
   relatório conta quantas caíram em cada teste.

O passo 3 é o que separa um risco de verdade de um fio de cabelo ou da dobra
de um vestido. Para ver o relatório::

    python -m retrato_vivo restaurar foto.jpg --explicar

Todos os tamanhos são relativos à resolução (``s = maior_lado / 1000``),
então o mesmo ajuste vale para fotos de 600 px e de 4000 px.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import damage_ai
from .faces import Box, detect_faces, faces_region


# =============================================================== configuração

@dataclass
class DamageConfig:
    sensitivity: float = 0.5              # 0 = só o óbvio, 1 = pega tudo (mais falso positivo)
    detect_scratches: bool = True         # riscos finos
    detect_creases: bool = True           # vincos e rachaduras (linhas claras largas)
    detect_spots: bool = True
    detect_large_losses: bool = False     # rasgos/áreas brancas (desligado por padrão)
    protect_faces: bool = True
    use_ai: bool = True                   # soma o detector de riscos com IA (damage_ai.py), se baixado

    # --- riscos ---
    scratch_min_length: float = 0.025     # comprimento mínimo (fração do maior lado)
    scratch_min_elongation: float = 5.0   # quão "comprido e fino" precisa ser
    # --- manchas ---
    spot_max_size: float = 0.012          # diâmetro máximo (fração do maior lado)
    spot_cluster_window: float = 0.06     # tamanho da vizinhança para contar manchas próximas
    spot_max_neighbors: int = 2           # mais vizinhas que isso = padrão da cena, não poeira
    # --- comuns ---
    texture_max_density: float = 0.30     # acima disso a região é "textura" (cabelo, tecido)
    edge_ratio: float = 0.8               # tolerância a bordas da cena perto do candidato
    dilate_px: float = 1.5                # margem extra ao redor do dano (escala 1000 px)


@dataclass
class DamageResult:
    mask: np.ndarray                      # uint8 0/255: máscara final usada no reparo
    scratches: np.ndarray                 # bool: só os riscos
    creases: np.ndarray                   # bool: só os vincos
    spots: np.ndarray                     # bool: só as manchas
    losses: np.ndarray                    # bool: só as perdas grandes
    ai_scratches: np.ndarray | None = None  # bool: riscos achados pela IA (None = IA não rodou)
    faces: list[Box] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)
    # relatório: detector -> {teste: quantas candidatas ele descartou}
    report: dict[str, dict[str, int]] = field(default_factory=dict)


# ================================================================ utilidades

def _lerp(a: float, b: float, t: float) -> float:
    """Valor entre ``a`` (t=0) e ``b`` (t=1). Usado para a sensibilidade."""
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def _odd(n: float, minimum: int = 3) -> int:
    """Arredonda para ímpar (tamanhos de filtro precisam ser ímpares)."""
    n = max(minimum, int(round(n)))
    return n if n % 2 else n + 1


def _scale(shape: tuple[int, ...]) -> float:
    return max(shape[:2]) / 1000.0


def _disk(size: float) -> np.ndarray:
    """Elemento estruturante circular."""
    k = _odd(size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _contrast_threshold(resp: np.ndarray, sensitivity: float, strict: float, loose: float) -> float:
    """Contraste mínimo para um pixel contar como candidato.

    É o maior entre: (a) um piso fixo em níveis de cinza, de ``strict``
    (sensibilidade 0) a ``loose`` (sensibilidade 1); (b) um valor adaptado à
    foto: mediana + k desvios (estimados pelo MAD, robusto a outliers), para
    fotos granuladas não gerarem milhares de candidatos.
    """
    median = float(np.median(resp))
    sigma = 1.4826 * float(np.median(np.abs(resp.astype(np.float32) - median)))
    k = _lerp(10.0, 4.0, sensitivity)
    return max(_lerp(strict, loose, sensitivity), median + k * sigma)


def _scene_edges(gray: np.ndarray, remove_size: float, measure_size: float) -> np.ndarray:
    """Quanto a CENA varia em volta de cada pixel, ignorando coisas pequenas.

    Primeiro apaga tudo que é menor que ``remove_size`` (abertura +
    fechamento), o que elimina o próprio risco/mancha. Depois mede a
    diferença entre o máximo e o mínimo local (gradiente morfológico).
    Resultado alto = há uma borda de objeto ali (dobra de tecido, contorno).
    Dano de verdade é independente da cena e costuma dar valor baixo.
    """
    se = _disk(remove_size)
    scene = cv2.morphologyEx(cv2.morphologyEx(gray, cv2.MORPH_OPEN, se), cv2.MORPH_CLOSE, se)
    return cv2.morphologyEx(scene, cv2.MORPH_GRADIENT, _disk(measure_size)).astype(np.float64)


class Candidates:
    """Regiões candidatas a dano e o histórico de quais testes as descartaram.

    Nasce de duas máscaras, como no detector de bordas de Canny (histerese):
    ``weak`` (pixels com algum indício) é dividida em regiões conexas, e só
    ficam vivas as regiões que contêm ao menos um pixel ``strong`` (indício
    forte). Assim um risco com trechos fracos não é quebrado em pedaços.

    Os atributos são vetores com um valor POR REGIÃO (índice 0 = fundo).
    """

    def __init__(self, weak: np.ndarray, strong: np.ndarray):
        self.n, self.labels, stats, _ = cv2.connectedComponentsWithStats(
            weak.astype(np.uint8), connectivity=8)
        self.alive = np.zeros(self.n, dtype=bool)
        self.alive[np.unique(self.labels[strong & weak])] = True
        self.alive[0] = False

        self.x, self.y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
        self.width, self.height = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
        self.area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        self.length = np.hypot(self.width, self.height)            # diagonal da caixa
        self.elongation = self.length ** 2 / np.maximum(self.area, 1.0)  # ~ comprimento/largura
        self.fill = self.area / np.maximum(self.width * self.height, 1)  # quanto da caixa ocupa
        self.report: dict[str, int] = {"candidatos": int(self.alive.sum())}

    def reject(self, reason: str, failed: np.ndarray) -> None:
        """Descarta as regiões vivas em que ``failed`` é True e anota quantas foram."""
        self.report[reason] = int((self.alive & failed).sum())
        self.alive &= ~failed

    def mean_of(self, image: np.ndarray) -> np.ndarray:
        """Média de ``image`` dentro de cada região."""
        sums = np.bincount(self.labels.ravel(), weights=image.ravel().astype(np.float64),
                           minlength=self.n)
        return sums / np.maximum(self.area, 1.0)

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Centro (x, y) da caixa de cada região."""
        h, w = self.labels.shape
        return (np.clip(self.x + self.width // 2, 0, w - 1),
                np.clip(self.y + self.height // 2, 0, h - 1))

    def inside(self, region: np.ndarray | None) -> np.ndarray:
        """Para cada região: o centro dela cai dentro de ``region``?"""
        if region is None:
            return np.zeros(self.n, dtype=bool)
        cx, cy = self.centers()
        return region[cy, cx]

    def mask(self) -> np.ndarray:
        """Imagem booleana com as regiões que sobreviveram."""
        return self.alive[self.labels]

    def finish(self) -> np.ndarray:
        self.report["aceitos"] = int(self.alive.sum())
        return self.mask()


# ==================================================================== riscos

def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Elemento estruturante em forma de segmento de reta inclinado."""
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    dx, dy = np.cos(np.radians(angle_deg)) * c, np.sin(np.radians(angle_deg)) * c
    cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return k


def linear_response(resp: np.ndarray, length_frac: float, n_angles: int = 12,
                    work_side: int = 1000) -> np.ndarray:
    """Mantém só o que é RETO E LONGO.

    Faz uma abertura morfológica com um segmento de reta em ``n_angles``
    inclinações e fica com o máximo. A abertura só preserva o que "cabe" o
    segmento inteiro: riscos sim; poeira, textura e bordas curtas não.

    Para ser rápido, roda numa cópia reduzida feita por *max-pooling* (pega o
    máximo de cada bloco), que não apaga linhas finas como uma média faria.
    """
    H, W = resp.shape
    ws = min(1.0, work_side / max(H, W))
    pool = int(np.ceil(1.0 / ws))
    r = cv2.dilate(resp, np.ones((pool + 2, pool + 2), np.uint8))  # engrossa: tolera inclinações intermediárias
    if ws < 1.0:
        r = cv2.resize(r, (max(1, round(W * ws)), max(1, round(H * ws))),
                       interpolation=cv2.INTER_NEAREST)
    length = _odd(length_frac * max(r.shape), minimum=7)
    out = np.zeros_like(r)
    for angle in np.arange(n_angles) * (180.0 / n_angles):
        np.maximum(out, cv2.morphologyEx(r, cv2.MORPH_OPEN, _line_kernel(length, angle)), out=out)
    if ws < 1.0:
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)
    return out


@dataclass(frozen=True)
class LineKind:
    """Um tipo de dano em forma de linha. Riscos e vincos usam a mesma
    receita (``_detect_lines``); só mudam estes números."""

    max_width: float          # largura máxima da linha (escala 1000 px)
    dark_too: bool            # procura também linhas escuras?
    texture_window: float     # tamanho da janela que mede "textura" (escala 1000 px)
    contrast_strict: float    # contraste mínimo com sensibilidade 0
    contrast_loose: float     # contraste mínimo com sensibilidade 1
    white_margin: float = 0.0  # se > 0, a linha precisa ser quase tão clara quanto o branco da foto


# Risco: arranhão fino no filme ou no papel. Pode ser claro ou escuro.
SCRATCH = LineKind(max_width=7, dark_too=True, texture_window=31,
                   contrast_strict=40.0, contrast_loose=14.0)
# Vinco: dobra ou rachadura do papel. É mais largo e sempre claro (o papel
# aparece por baixo da emulsão). Como "largo e claro" também descreve muita
# coisa da cena (golas, frisos), exige mais contraste e que a linha seja
# quase branca: no máximo 60 níveis abaixo do ponto mais claro da foto.
CREASE = LineKind(max_width=15, dark_too=False, texture_window=61,
                  contrast_strict=55.0, contrast_loose=22.0, white_margin=60.0)


def _detect_lines(gray: np.ndarray, cfg: DamageConfig, kind: LineKind,
                  face_region: np.ndarray | None) -> tuple[np.ndarray, dict[str, int]]:
    """Receita comum de riscos e vincos. Retorna ``(máscara, relatório)``."""
    s = _scale(gray.shape)
    max_side = max(gray.shape)
    thin = _disk(kind.max_width * s)   # "fino" = mais estreito que este disco

    # 1. REALÇAR o que é fino: top-hat pega o fino e claro, black-hat o fino e escuro.
    thinness = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, thin)
    if kind.dark_too:
        thinness = np.maximum(thinness, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, thin))

    #    ...e, do que é fino, o que também é reto e longo.
    linear = linear_response(thinness, cfg.scratch_min_length * 1.5)

    #    Regiões cheias de coisas finas (cabelo, plumas, tecido) são textura,
    #    não dano. São zeradas já aqui, senão um risco que cruza o cabelo
    #    se funde com ele numa região gigante. A janela precisa ser bem maior
    #    que a largura da linha, senão a própria linha conta como textura.
    contrast = _contrast_threshold(thinness, cfg.sensitivity,
                                   kind.contrast_strict, kind.contrast_loose)
    is_thin = thinness >= 0.5 * contrast
    density = cv2.blur(is_thin.astype(np.float32), (_odd(kind.texture_window * s),) * 2)
    texture = density > cfg.texture_max_density

    # 2. AGRUPAR: sementes fortes onde é reto e longo; crescem pelos pixels finos vizinhos.
    near_line = cv2.dilate((linear >= 0.5 * contrast).astype(np.uint8),
                           np.ones((3, 3), np.uint8)).astype(bool)
    c = Candidates(weak=is_thin & near_line & ~texture,
                   strong=(linear >= contrast) & ~texture)

    # 3. FILTRAR. Dentro de rostos os critérios são 1,5x mais exigentes.
    strictness = np.where(c.inside(face_region), 1.5, 1.0)
    c.reject("curto demais", c.length < cfg.scratch_min_length * max_side * strictness)
    if kind.white_margin > 0:
        paper_white = np.percentile(gray, 99.5)
        c.reject("não é branco", c.mean_of(gray) < paper_white - kind.white_margin)
    c.reject("pouco alongado", c.elongation < cfg.scratch_min_elongation * strictness)

    #    Dobra de tecido, friso de carro e contorno são linhas finas que ficam
    #    SOBRE uma borda da cena. Um risco não depende da cena.
    scene_edge = c.mean_of(_scene_edges(gray, remove_size=kind.max_width * s,
                                        measure_size=(kind.max_width + 4) * s))
    own_contrast = c.mean_of(thinness)
    tolerance = _lerp(0.7, 1.3, cfg.sensitivity) * cfg.edge_ratio
    c.reject("em borda da cena", scene_edge > tolerance * own_contrast)

    return c.finish(), c.report


def detect_scratches(gray: np.ndarray, cfg: DamageConfig,
                     face_region: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """Encontra riscos finos, claros ou escuros."""
    return _detect_lines(gray, cfg, SCRATCH, face_region)


def detect_creases(gray: np.ndarray, cfg: DamageConfig,
                   face_region: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """Encontra vincos e rachaduras: linhas claras mais largas que um risco."""
    return _detect_lines(gray, cfg, CREASE, face_region)


# =================================================================== manchas

def _surroundings_detail(c: Candidates, gray: np.ndarray, s: float, max_d: float) -> np.ndarray:
    """Para cada candidata viva, o quanto de detalhe existe num ANEL em volta dela.

    Poeira cai sobre fundo liso (anel sem detalhe). Pupila, botão e textura
    têm detalhe também ao redor.
    """
    blob = c.mask().astype(np.uint8)
    gap = _disk(2 * s + 1)                                  # folga entre a mancha e o anel
    reach = _disk(max(5.0, 0.5 * max_d) * 2 + gap.shape[0])  # até onde o anel vai
    inner = cv2.dilate(blob, gap)
    # Dilatar a imagem de rótulos "espalha" o número de cada mancha pelo anel dela.
    spread = cv2.dilate(c.labels.astype(np.float32) * blob, reach)
    ring = (spread > 0) & (inner == 0)
    detail = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), max(1.0, 1.5 * s))).astype(np.float64)
    owner = spread[ring].astype(np.int64)
    total = np.bincount(owner, weights=detail[ring], minlength=c.n)
    count = np.bincount(owner, minlength=c.n)
    return total / np.maximum(count, 1)


def _neighbors(c: Candidates, window: int) -> np.ndarray:
    """Para cada candidata viva, quantas OUTRAS candidatas vivas há por perto."""
    result = np.zeros(c.n, dtype=np.float64)
    ids = np.flatnonzero(c.alive)
    if ids.size == 0:
        return result
    cx, cy = c.centers()
    cx, cy = cx[ids], cy[ids]
    centers = np.zeros(c.labels.shape, np.float32)
    np.add.at(centers, (cy, cx), 1.0)
    counts = cv2.boxFilter(centers, -1, (window, window), normalize=False,
                           borderType=cv2.BORDER_CONSTANT)
    result[ids] = counts[cy, cx] - 1
    return result


def detect_spots(gray: np.ndarray, cfg: DamageConfig,
                 face_region: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """Encontra poeira e manchas pequenas, claras ou escuras. Retorna ``(máscara, relatório)``."""
    s = _scale(gray.shape)
    max_d = max(3.0, cfg.spot_max_size * max(gray.shape))   # maior diâmetro aceito, em pixels

    # 1. REALÇAR o que difere da vizinhança: a mediana local "apaga" tudo que
    #    é pequeno, então |foto - mediana| deixa só o que é pequeno.
    median = cv2.medianBlur(gray, min(_odd(2 * max_d + 1), 255))
    difference = cv2.absdiff(gray, median)

    # 2. AGRUPAR.
    contrast = _contrast_threshold(difference, cfg.sensitivity, strict=60.0, loose=25.0)
    c = Candidates(weak=difference >= 0.6 * contrast, strong=difference >= contrast)

    # 3. FILTRAR.
    in_face = c.inside(face_region)
    max_area = np.pi * (max_d / 2.0) ** 2 * 1.5 * np.where(in_face, 0.4, 1.0)
    c.reject("tamanho errado", (c.area < 2) | (c.area > max_area))
    c.reject("não é compacta", (c.fill < 0.3) | (c.elongation > 6.0))

    own_contrast = c.mean_of(difference)
    max_detail = _lerp(0.12, 0.25, cfg.sensitivity) * np.where(in_face, 0.5, 1.0)
    c.reject("entorno com detalhe",
             _surroundings_detail(c, gray, s, max_d) > max_detail * own_contrast)

    scene_edge = c.mean_of(_scene_edges(gray, remove_size=max_d + 2, measure_size=max_d * 1.5 + 2))
    tolerance = _lerp(0.8, 1.4, cfg.sensitivity) * cfg.edge_ratio
    c.reject("em borda da cena", scene_edge > tolerance * own_contrast)

    #    Poeira é esparsa e aleatória. Janelas de prédio, pedras de muro e
    #    botões aparecem em grupos.
    window = _odd(cfg.spot_cluster_window * max(gray.shape))
    c.reject("padrão repetido", _neighbors(c, window) > cfg.spot_max_neighbors)

    return c.finish(), c.report


# ============================================================ perdas grandes

def detect_large_losses(gray: np.ndarray, cfg: DamageConfig) -> tuple[np.ndarray, dict[str, int]]:
    """Encontra áreas grandes, quase brancas e sem textura (rasgo, papel faltando).

    Desligado por padrão porque confunde com céu estourado e paredes brancas.
    """
    s = _scale(gray.shape)
    g = gray.astype(np.float32)
    win = (_odd(9 * s),) * 2
    local_variance = cv2.blur(g * g, win) - cv2.blur(g, win) ** 2

    very_bright = gray >= _lerp(250, 232, cfg.sensitivity)
    flat = local_variance < 30.0
    cand = cv2.morphologyEx((very_bright & flat).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)).astype(bool)

    c = Candidates(weak=cand, strong=cand)
    c.reject("pequena demais", c.area < 0.0015 * gray.shape[0] * gray.shape[1])
    return c.finish(), c.report


# =============================================================== API pública

def detect_damage(img: np.ndarray, cfg: DamageConfig | None = None,
                  mask_add: np.ndarray | None = None,
                  mask_remove: np.ndarray | None = None) -> DamageResult:
    """Roda todos os detectores ligados e monta a máscara final de reparo.

    ``mask_add`` e ``mask_remove`` (mesmo tamanho da imagem) vêm do editor
    da GUI: o usuário pinta o que faltou e apaga o que foi marcado por engano.
    """
    cfg = cfg or DamageConfig()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nothing = np.zeros(gray.shape, dtype=bool)
    report: dict[str, dict[str, int]] = {}

    faces = detect_faces(img) if cfg.protect_faces else []
    face_region = faces_region(gray.shape, faces) if faces else None

    scratches = creases = spots = losses = nothing
    if cfg.detect_scratches:
        scratches, report["riscos"] = detect_scratches(gray, cfg, face_region)
    if cfg.detect_creases:
        creases, report["vincos"] = detect_creases(gray, cfg, face_region)
    if cfg.detect_spots:
        spots, report["manchas"] = detect_spots(gray, cfg, face_region)
    if cfg.detect_large_losses:
        losses, report["perdas"] = detect_large_losses(gray, cfg)

    # Junta tudo e dá uma pequena margem: o dano costuma ter uma "auréola".
    ai = None
    if cfg.use_ai and damage_ai.is_available():
        ai, report["riscos_ia"] = damage_ai.detect_scratches_ai(img, cfg.sensitivity)

    mask = (scratches | creases | spots | losses | (ai if ai is not None else nothing)).astype(np.uint8) * 255
    if cfg.dilate_px > 0 and mask.any():
        mask = cv2.dilate(mask, _disk(2 * cfg.dilate_px * _scale(gray.shape) + 1))
    if mask_add is not None:
        mask[np.asarray(mask_add) > 0] = 255
    if mask_remove is not None:
        mask[np.asarray(mask_remove) > 0] = 0

    stats = {
        "area_danificada_pct": 100.0 * float((mask > 0).mean()),
        "riscos": float(report.get("riscos", {}).get("aceitos", 0)),
        "vincos": float(report.get("vincos", {}).get("aceitos", 0)),
        "riscos_ia": float(report.get("riscos_ia", {}).get("aceitos", 0)),
        "manchas": float(report.get("manchas", {}).get("aceitos", 0)),
        "rostos_protegidos": float(len(faces)),
    }
    return DamageResult(mask=mask, scratches=scratches, creases=creases, spots=spots, losses=losses,
                        ai_scratches=ai, faces=faces, stats=stats, report=report)
