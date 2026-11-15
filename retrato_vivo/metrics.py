"""Métricas objetivas de qualidade (para o relatório/apresentação)."""

from __future__ import annotations

import cv2
import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio em dB (maior = mais parecido)."""
    return float(cv2.PSNR(a, b))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity (Wang et al., 2004) em tons de cinza, janela
    gaussiana 11x11, sigma 1.5. Retorna valor em [-1, 1] (1 = idênticas)."""
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def g(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = g(a), g(b)
    var_a = g(a * a) - mu_a ** 2
    var_b = g(b * b) - mu_b ** 2
    cov = g(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float((num / den).mean())


def mask_scores(pred: np.ndarray, truth: np.ndarray, tolerance_px: int = 2) -> dict[str, float]:
    """Precisão, revocação, F1 e IoU entre a máscara prevista e a verdadeira.

    ``tolerance_px`` aceita pequenos desvios de posição (bordas
    antisserrilhadas), dilatando a máscara de referência ao contar acertos.
    """
    p, t = np.asarray(pred) > 0, np.asarray(truth) > 0
    k = np.ones((2 * tolerance_px + 1,) * 2, np.uint8)
    t_dil = cv2.dilate(t.astype(np.uint8), k).astype(bool)
    p_dil = cv2.dilate(p.astype(np.uint8), k).astype(bool)
    precision = float((p & t_dil).sum() / max(p.sum(), 1))
    recall = float((t & p_dil).sum() / max(t.sum(), 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    iou = float((p & t).sum() / max((p | t).sum(), 1))
    return {"precisao": precision, "revocacao": recall, "f1": f1, "iou": iou}
