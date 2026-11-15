"""Pré-processamento clássico: cor, contraste, ruído e nitidez (OpenCV puro).

Regra de ouro: NÃO ESTRAGAR FOTO BOA. Por isso tudo começa por um
diagnóstico (``diagnose``), e cada correção só age se o diagnóstico mostrar
que a foto precisa, na dose que ela precisa. Cada decisão vira uma frase em
``StepLog.notes`` (ex.: "ruído medido 0,7: foto limpa, etapa pulada").

As correções rodam em dois momentos, porque a ordem importa:

- ``normalize``: ANTES de procurar danos. Tira a dominante de cor e
  recupera o contraste, o que deixa a detecção estável entre fotos.
- ``enhance``: DEPOIS de apagar os danos. Tira o ruído, realça o contraste
  local e a nitidez. Se a nitidez viesse antes, realçaria os riscos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# =============================================================== configuração

@dataclass
class PreprocessConfig:
    # --- normalize (antes do reparo) ---
    remove_color_cast: bool = True   # tira amarelado/sépia
    auto_levels: bool = True         # recupera preto e branco de foto desbotada
    # --- enhance (depois do reparo) ---
    denoise: bool = True
    denoise_strength: float = 1.0    # multiplica a dose automática (0.5 = metade, 2 = dobro)
    clahe: bool = True
    clahe_strength: float = 0.4      # 0 = nada, 1 = CLAHE puro
    sharpen: bool = True
    sharpen_amount: float = 0.5


@dataclass
class StepLog:
    """Guarda a imagem de cada etapa aplicada e a explicação de cada decisão."""

    steps: dict[str, np.ndarray] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, img: np.ndarray, note: str = "") -> np.ndarray:
        self.steps[name] = img
        if note:
            self.notes[name] = note
        return img

    def skip(self, name: str, reason: str) -> None:
        self.notes[name] = f"pulada: {reason}"


# ================================================================ diagnóstico

@dataclass
class Diagnosis:
    """O que foi medido na foto original."""

    monochrome: bool                 # P&B ou sépia (sem variação de cor)
    chroma_std: float                # quanto a cor varia entre pixels
    cast_ab: tuple[float, float]     # dominante de cor estimada, nos eixos a/b do LAB
    cast_ref_l: float                # luminosidade média dos pixels usados na estimativa
    noise_sigma: float               # ruído estimado, em níveis de cinza
    black_point: float               # L do percentil 0,5
    white_point: float               # L do percentil 99,5

    @property
    def cast_strength(self) -> float:
        return float(np.hypot(*self.cast_ab))


MONOCHROME_MAX_CHROMA_STD = 6.0   # abaixo disso a foto é tratada como P&B/sépia
MIN_CAST_TO_CORRECT = 2.0         # dominante menor que isso é ignorada
MAX_CAST_CORRECTION = 15.0        # teto de segurança da correção
NOISE_TO_DENOISE = 1.5            # ruído abaixo disso = foto limpa
LEVELS_CLIP_PCT = 0.5             # percentis usados como ponto preto/branco


def estimate_noise(img: np.ndarray) -> float:
    """Estima o desvio padrão do ruído, em níveis de cinza.

    Aplica um filtro que cancela regiões lisas e gradientes suaves (sobra
    quase só ruído) e usa a mediana do resultado, que ignora as bordas da
    cena. Foto digital limpa dá menos de 1; filme granulado, de 3 a 10.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    response = cv2.filter2D(gray, -1, kernel)
    return float(np.median(np.abs(response)) / 0.6745 / 6.0)


def _estimate_cast(lab: np.ndarray) -> tuple[tuple[float, float], float]:
    """Estima a dominante de cor de uma foto colorida.

    Ideia: toda foto sem dominante tem ALGUM pixel neutro (um branco, um
    cinza). Se nem os pixels mais neutros da foto são neutros, a cor deles é
    a dominante. Usa o 1% menos colorido entre os pixels não escuros (nas
    sombras a dominante quase não aparece).
    """
    L, a, b = lab[:, :, 0], lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
    bright_enough = L >= 60
    if bright_enough.sum() < 500:
        return (0.0, 0.0), 128.0
    chroma = np.hypot(a, b)[bright_enough]
    most_neutral = chroma <= np.percentile(chroma, 1)
    cast = (float(a[bright_enough][most_neutral].mean()),
            float(b[bright_enough][most_neutral].mean()))
    return cast, float(L[bright_enough][most_neutral].mean())


def diagnose(img: np.ndarray) -> Diagnosis:
    """Mede cor, ruído e contraste da foto. Não altera nada."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0

    # Foto sépia tem a cor DESLOCADA mas quase sem VARIAÇÃO entre pixels;
    # por isso o critério de "monocromática" é o desvio padrão, não a média.
    chroma_std = float(np.hypot(a.std(), b.std()))
    monochrome = chroma_std < MONOCHROME_MAX_CHROMA_STD
    if monochrome:
        cast, ref_l = (float(a.mean()), float(b.mean())), float(lab[:, :, 0].mean())
    else:
        cast, ref_l = _estimate_cast(lab)

    black, white = np.percentile(lab[:, :, 0], (LEVELS_CLIP_PCT, 100.0 - LEVELS_CLIP_PCT))
    return Diagnosis(monochrome=monochrome, chroma_std=chroma_std, cast_ab=cast,
                     cast_ref_l=ref_l, noise_sigma=estimate_noise(img),
                     black_point=float(black), white_point=float(white))


# ================================================================== normalize

def is_gray(img: np.ndarray) -> bool:
    """True se os três canais são idênticos (dá para processar um só)."""
    return bool(np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]))


def _from_lab(lab: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Volta de LAB para BGR. Se a foto de origem era cinza, garante que o
    resultado continua exatamente cinza (a conversão erra por 1 nível)."""
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return to_neutral_gray(out) if is_gray(source) else out


def to_neutral_gray(img: np.ndarray) -> np.ndarray:
    """Foto P&B/sépia vira cinza neutro. A cor "real" é recriada depois pela
    colorização com Deep Learning."""
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def remove_color_cast(img: np.ndarray, cast_ab: tuple[float, float], ref_l: float) -> np.ndarray:
    """Subtrai a dominante de cor de uma foto colorida.

    A correção é proporcional à luminosidade (o amarelado do papel aparece
    mais nos tons claros que nas sombras) e tem um teto de segurança.
    """
    strength = float(np.hypot(*cast_ab))
    limit = min(1.0, MAX_CAST_CORRECTION / max(strength, 1e-6))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    weight = np.clip(lab[:, :, 0] / max(ref_l, 1.0), 0.0, 1.5) * limit
    lab[:, :, 1] -= cast_ab[0] * weight
    lab[:, :, 2] -= cast_ab[1] * weight
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def auto_levels(img: np.ndarray, black_target: int = 4, white_target: int = 240,
                max_gain: float = 1.5) -> tuple[np.ndarray, float]:
    """Recupera o preto e o branco de uma foto desbotada. Retorna ``(imagem, ganho)``.

    Só EXPANDE o contraste, nunca comprime: se o preto da foto já é preto e
    o branco já é branco, o ganho é 1 e nada muda. O alvo do branco fica
    abaixo de 255 e o ganho tem teto, para o fundo claro de um retrato não
    "estourar". Age só no canal L (LAB), sem alterar a cor.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    black, white = np.percentile(L, (LEVELS_CLIP_PCT, 100.0 - LEVELS_CLIP_PCT))
    if white - black < 10:               # imagem praticamente uniforme
        return img, 1.0
    new_black, new_white = min(black, black_target), max(white, white_target)
    gain = (new_white - new_black) / (white - black)
    if gain <= 1.03:
        return img, 1.0
    if gain > max_gain:                  # no teto, expande em torno do meio
        gain = max_gain
        new_black = (black + white) / 2.0 * (1 - gain) + black * gain
    lut = np.clip((np.arange(256, dtype=np.float32) - black) * gain + new_black, 0, 255)
    lab[:, :, 0] = cv2.LUT(L, lut.astype(np.uint8))
    return _from_lab(lab, img), float(gain)


def normalize(img: np.ndarray, cfg: PreprocessConfig, diag: Diagnosis,
              log: StepLog | None = None) -> np.ndarray:
    """Correções feitas ANTES da detecção de danos."""
    log = log or StepLog()

    if cfg.remove_color_cast:
        if diag.monochrome:
            img = log.add("cor_neutralizada", to_neutral_gray(img),
                          "foto P&B/sépia: convertida para cinza neutro")
        elif diag.cast_strength >= MIN_CAST_TO_CORRECT:
            img = log.add("cor_neutralizada", remove_color_cast(img, diag.cast_ab, diag.cast_ref_l),
                          f"dominante de cor {diag.cast_strength:.1f} removida")
        else:
            log.skip("cor_neutralizada", f"sem dominante de cor ({diag.cast_strength:.1f})")

    if cfg.auto_levels:
        leveled, gain = auto_levels(img)
        if gain > 1.0:
            img = log.add("niveis_automaticos", leveled, f"contraste expandido {gain:.2f}x")
        else:
            log.skip("niveis_automaticos", "a foto já usa toda a faixa de tons")
    return img


# ==================================================================== enhance

def denoise(img: np.ndarray, h: float) -> np.ndarray:
    """Remove granulação com Non-Local Means: troca cada pixel pela média de
    trechos PARECIDOS da vizinhança, o que preserva as bordas melhor que um
    desfoque. ``h`` é a força. Em foto cinza processa um canal só (3x mais rápido)."""
    if is_gray(img):
        g = cv2.fastNlMeansDenoising(img[:, :, 0], None, h=h,
                                     templateWindowSize=7, searchWindowSize=21)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return cv2.fastNlMeansDenoisingColored(img, None, h=h, hColor=h,
                                           templateWindowSize=7, searchWindowSize=21)


def denoise_dose(noise_sigma: float, strength: float = 1.0) -> float:
    """Força do Non-Local Means para o ruído medido (0 = não precisa).

    Em foto sem ruído o filtro só atrapalha: cria "degraus" em céus e fundos
    suaves e deixa a pele com aspecto de plástico.
    """
    if noise_sigma < NOISE_TO_DENOISE:
        return 0.0
    return float(np.clip(0.6 * noise_sigma * strength, 1.0, 12.0))


def apply_clahe(img: np.ndarray, strength: float = 0.4, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Realça o contraste LOCAL (CLAHE) no canal L: recupera detalhe em áreas
    escuras e claras. ``strength`` mistura o resultado com a foto original,
    porque o CLAHE puro deixa a foto artificial."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    equalized = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(L)
    lab[:, :, 0] = cv2.addWeighted(equalized, strength, L, 1.0 - strength, 0)
    return _from_lab(lab, img)


def unsharp_mask(img: np.ndarray, amount: float = 0.5, sigma: float = 1.2) -> np.ndarray:
    """Nitidez: soma à foto a diferença entre ela e uma cópia desfocada."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def enhance(img: np.ndarray, cfg: PreprocessConfig, log: StepLog | None = None) -> np.ndarray:
    """Correções feitas DEPOIS do reparo de danos."""
    log = log or StepLog()

    if cfg.denoise:
        noise = estimate_noise(img)   # medido de novo: os níveis podem ter amplificado o ruído
        h = denoise_dose(noise, cfg.denoise_strength)
        if h > 0:
            img = log.add("ruido_removido", denoise(img, h),
                          f"ruído medido {noise:.1f}: Non-Local Means com força {h:.1f}")
        else:
            log.skip("ruido_removido", f"ruído medido {noise:.1f}: foto limpa")

    if cfg.clahe and cfg.clahe_strength > 0:
        img = log.add("clahe", apply_clahe(img, cfg.clahe_strength),
                      f"contraste local realçado ({cfg.clahe_strength:.0%})")
    if cfg.sharpen and cfg.sharpen_amount > 0:
        img = log.add("nitidez", unsharp_mask(img, cfg.sharpen_amount),
                      f"nitidez {cfg.sharpen_amount:.0%}")
    return img
