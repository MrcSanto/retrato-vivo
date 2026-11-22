"""Correção de perspectiva: "foto de uma foto" tirada com o celular.

Quem fotografa uma foto impressa com o celular quase nunca fica bem de frente:
o papel sai torto, em forma de trapézio, e com a mesa em volta. Esta etapa acha
as quatro bordas do papel, "desentorta" e recorta, como um scanner faria.

Como funciona:

1. **Achar o papel** numa cópia reduzida (rápido), por dois caminhos, porque
   nenhum funciona sempre:

   - *Bordas de Canny*, engrossadas por dilatação para fechar falhas: acha o
     papel quando a borda dele é nítida, mesmo em fundo parecido.
   - *Limiarização de Otsu*: separa o papel (claro) do fundo (escuro), ou o
     contrário. Acha o papel quando a borda é borrada, mas o contraste é bom.

   De cada caminho saem contornos (``findContours``). O fecho convexo de cada um
   é simplificado por ``approxPolyDP``; os que viram exatamente 4 cantos são
   candidatos.
2. **Desconfiar.** Um candidato só vale se for convexo, ocupar de 15% a 95% da
   foto (menos que isso é um objeto qualquer; mais é a própria foto, sem fundo
   em volta) e tiver ângulos entre 45° e 135° (um trapézio de perspectiva, não
   uma forma qualquer). Fica o maior candidato válido. Sem nenhum, a foto volta
   **intacta**: uma foto digitalizada normal não pode ser estragada.
3. **Desentortar.** Os cantos são ordenados (sup.-esq., sup.-dir., inf.-dir.,
   inf.-esq.), ``getPerspectiveTransform`` calcula a homografia que leva o
   trapézio para um retângulo, e ``warpPerspective`` aplica. A largura e a
   altura do retângulo são os maiores lados opostos do trapézio.
"""

from __future__ import annotations

import cv2
import numpy as np

DETECT_SIDE = 800           # a busca roda numa cópia com este lado maior
MIN_AREA = 0.15             # fração da foto: abaixo disso não é o papel
MAX_AREA = 0.95             # acima disso é a própria foto, sem fundo em volta
MIN_ANGLE, MAX_ANGLE = 45.0, 135.0
MAX_CONVERGENCE = 30.0      # graus entre lados opostos: a perspectiva aproxima, mas pouco
BORDER_MARGIN = 0.01        # fração do lado maior: canto mais perto da beira que isso não vale
MIN_SIDE_SUPPORT = 0.5      # cada lado precisa coincidir com bordas da foto em pelo menos metade
SIDE_SAMPLES = 60
INSET = 0.004               # puxa os cantos um pouco para dentro: tira o fio de mesa da borda
APPROX_EPSILONS = (0.02, 0.03, 0.05)   # tolerâncias do approxPolyDP, da mais fiel à mais solta


def ordenar_cantos(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos como sup.-esq., sup.-dir., inf.-dir., inf.-esq. (float32, 4×2).

    O canto sup.-esq. tem a menor soma x+y e o inf.-dir., a maior; o sup.-dir.
    tem a menor diferença y-x e o inf.-esq., a maior.
    """
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def _angles(quad: np.ndarray) -> np.ndarray:
    """Ângulo interno, em graus, de cada canto do quadrilátero."""
    out = []
    for i in range(4):
        a, b, c = quad[i - 1], quad[i], quad[(i + 1) % 4]
        u, v = a - b, c - b
        cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
        out.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.array(out)


def _side_support(quad: np.ndarray, edges: np.ndarray) -> float:
    """Menor fração, entre os 4 lados, dos pontos do lado que caem sobre uma borda da foto."""
    h, w = edges.shape
    worst = 1.0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        t = np.linspace(0.05, 0.95, SIDE_SAMPLES)[:, None]     # as pontas ficam de fora: cantos arredondados
        pts = np.round(a + (b - a) * t).astype(int)
        xs, ys = np.clip(pts[:, 0], 0, w - 1), np.clip(pts[:, 1], 0, h - 1)
        worst = min(worst, float((edges[ys, xs] > 0).mean()))
    return worst


def _opposite_sides_angle(quad: np.ndarray) -> float:
    """Maior ângulo, em graus, entre lados opostos (0 = paralelos)."""
    worst = 0.0
    for i in (0, 1):
        u, v = quad[i + 1] - quad[i], quad[i + 2] - quad[(i + 3) % 4]   # lado i e o oposto, no mesmo sentido
        cos = abs(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
        worst = max(worst, float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))))
    return worst


def _is_valid(quad: np.ndarray, edges: np.ndarray) -> bool:
    """Os testes de desconfiança: só passa o que parece mesmo um papel fotografado sobre um fundo."""
    h, w = edges.shape
    if not cv2.isContourConvex(quad.reshape(4, 1, 2).astype(np.int32)):
        return False
    if not MIN_AREA <= cv2.contourArea(quad) / (h * w) <= MAX_AREA:
        return False
    margin = BORDER_MARGIN * max(h, w)              # fundo em volta: nenhum canto encosta na beira
    if (quad[:, 0] < margin).any() or (quad[:, 0] > w - 1 - margin).any() \
            or (quad[:, 1] < margin).any() or (quad[:, 1] > h - 1 - margin).any():
        return False
    angles = _angles(quad)
    if not ((angles >= MIN_ANGLE) & (angles <= MAX_ANGLE)).all():
        return False
    if _opposite_sides_angle(quad) > MAX_CONVERGENCE:
        return False
    return _side_support(quad, edges) >= MIN_SIDE_SUPPORT


def _masks(gray: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """As bordas da foto (engrossadas) e as imagens binárias onde o contorno do papel pode aparecer."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blur))
    edges = cv2.Canny(blur, int(max(0, 0.66 * median)), int(min(255, 1.33 * median)) + 1)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    otsu = cv2.morphologyEx(cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    return edges, [edges, otsu, 255 - otsu]


def encontrar_papel(img: np.ndarray) -> np.ndarray | None:
    """Os 4 cantos do papel na foto (float32, 4×2, ordenados), ou ``None`` se não houver um confiável."""
    h, w = img.shape[:2]
    k = min(1.0, DETECT_SIDE / max(h, w))
    small = img if k == 1.0 else cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges, masks = _masks(gray)

    best, best_area = None, 0.0
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            for eps in APPROX_EPSILONS:
                approx = cv2.approxPolyDP(hull, eps * perimeter, True)
                if len(approx) != 4:
                    continue
                quad = ordenar_cantos(approx)
                if _is_valid(quad, edges) and cv2.contourArea(quad) > best_area:
                    best, best_area = quad, cv2.contourArea(quad)
                break
    if best is None:
        return None
    center = best.mean(axis=0)
    best = best + (center - best) * INSET * 2      # um fio para dentro
    return (best / k).astype(np.float32)


def retificar(img: np.ndarray, cantos: np.ndarray) -> np.ndarray:
    """Leva o quadrilátero ``cantos`` para um retângulo (a foto desentortada e recortada)."""
    tl, tr, br, bl = ordenar_cantos(cantos)
    width = round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), target)
    return cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def desenhar_cantos(img: np.ndarray, cantos: np.ndarray) -> np.ndarray:
    """A foto com o contorno do papel encontrado (verde) e os cantos marcados."""
    out = img.copy()
    thickness = max(2, round(max(img.shape[:2]) / 300))
    pts = np.round(cantos).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, (0, 200, 0), thickness, cv2.LINE_AA)
    for x, y in pts.reshape(-1, 2):
        cv2.circle(out, (int(x), int(y)), thickness * 3, (0, 200, 0), -1, cv2.LINE_AA)
    return out


def corrigir_perspectiva(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Desentorta e recorta a foto impressa; devolve a foto e se achou o papel.

    Sem um papel confiável, devolve a própria ``img`` intacta e ``False``.
    """
    cantos = encontrar_papel(img)
    if cantos is None:
        return img, False
    return retificar(img, cantos), True
