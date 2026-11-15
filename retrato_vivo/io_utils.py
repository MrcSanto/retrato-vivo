"""Leitura, escrita e normalização de imagens.

Todo o projeto trabalha com o mesmo formato interno: ``numpy.ndarray`` BGR,
``uint8``, 3 canais. As funções daqui garantem essa conversão para qualquer
arquivo que o usuário (ou o professor) abrir: tons de cinza, PNG com
transparência, TIFF 16 bits, caminhos com acentos no Windows etc.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jp2", ".pgm", ".ppm",
)


def to_bgr8(img: np.ndarray) -> np.ndarray:
    """Converte qualquer imagem lida pelo OpenCV para BGR uint8 com 3 canais."""
    if img is None:
        raise ValueError("imagem vazia")

    # Profundidade -> uint8
    if img.dtype == np.uint16:
        img = (img / 257.0).round().astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        mx = float(img.max()) if img.size else 1.0
        scale = 255.0 if mx <= 1.0 else 1.0
        img = np.clip(img * scale, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Canais -> 3
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    channels = img.shape[2]
    if channels == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        # Compõe sobre fundo branco (papel fotográfico) em vez de descartar o alfa.
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        return (bgr * alpha + 255.0 * (1.0 - alpha)).round().astype(np.uint8)
    return img[:, :, :3].copy()


def load_image(path: str | Path) -> np.ndarray:
    """Lê uma imagem do disco como BGR uint8.

    Usa ``np.fromfile`` + ``cv2.imdecode`` porque ``cv2.imread`` falha com
    caminhos contendo acentos no Windows (ex.: ``C:/Fotos/Vovó.jpg``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"arquivo não encontrado: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)   # mantém transparência e 16 bits
    has_alpha = img is not None and img.ndim == 3 and img.shape[2] == 4
    if img is not None and not has_alpha:
        # IMREAD_UNCHANGED ignora a rotação gravada no EXIF: foto de celular
        # tirada em pé viria deitada. Este modo aplica a rotação.
        oriented = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        img = oriented if oriented is not None else img
    if img is None:
        img = _load_with_pillow(path)
    if img is None:
        raise ValueError(f"formato não suportado ou arquivo corrompido: {path}")
    return to_bgr8(img)


def _load_with_pillow(path: Path) -> np.ndarray | None:
    """Alternativa para formatos que o OpenCV não decodifica (AVIF, HEIC...).

    O OpenCV ignora a extensão e olha o conteúdo, mas não tem decodificador
    de AVIF. Imagens salvas do Google muitas vezes são AVIF com extensão
    ``.jpg``. O Pillow (>= 11.3) lê AVIF nativamente.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            arr = np.asarray(im)
    except Exception:  # formato desconhecido também para o Pillow
        return None
    code = cv2.COLOR_RGBA2BGRA if arr.shape[2] == 4 else cv2.COLOR_RGB2BGR
    return cv2.cvtColor(arr, code)


def save_image(path: str | Path, img: np.ndarray, quality: int = 95) -> Path:
    """Salva a imagem (aceita caminhos com acentos). Sem extensão, usa PNG."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, buf = cv2.imencode(path.suffix, img, params)
    if not ok:
        raise ValueError(f"não foi possível codificar a imagem como {path.suffix}")
    buf.tofile(str(path))
    return path


def resize_max_side(img: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
    """Reduz a imagem para que o maior lado tenha no máximo ``max_side`` pixels.

    Retorna ``(imagem, escala)``. Nunca amplia (ampliação é papel da
    super-resolução). ``max_side=None`` desativa.
    """
    h, w = img.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / float(max(h, w))
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale


def list_images(folder: str | Path) -> list[Path]:
    """Lista as imagens suportadas de uma pasta (não recursivo), em ordem alfabética."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
