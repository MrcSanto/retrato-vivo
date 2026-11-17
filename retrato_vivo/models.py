"""Registro central dos modelos pré-treinados (pesos ficam fora do git, em ``models/``).

Para adicionar um modelo novo, inclua uma entrada em ``MODELS`` e use
``model_path("nome")`` no código. Baixar tudo::

    python -m retrato_vivo modelos           # baixa todos
    python -m retrato_vivo modelos yunet     # baixa só um
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# nome -> (arquivo, url, descrição)
MODELS: dict[str, tuple[str, str, str]] = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx",
        "Detector de rostos YuNet (protege rostos na detecção de danos)",
    ),
    # Colorização de Zhang et al. (ECCV 2016), em Caffe: precisa dos três arquivos.
    "colorizacao_rede": (
        "colorization_deploy_v2.prototxt",
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/"
        "models/colorization_deploy_v2.prototxt",
        "Colorização (Zhang et al., ECCV 2016): arquitetura da rede",
    ),
    "colorizacao_pesos": (
        "colorization_release_v2.caffemodel",
        "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1",
        "Colorização (Zhang et al., ECCV 2016): pesos treinados (129 MB)",
    ),
    "colorizacao_cores": (
        "pts_in_hull.npy",
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/"
        "resources/pts_in_hull.npy",
        "Colorização (Zhang et al., ECCV 2016): as 313 cores que a rede escolhe",
    ),
}


def model_path(name: str) -> Path:
    return MODELS_DIR / MODELS[name][0]


def is_available(name: str) -> bool:
    return model_path(name).is_file()


def download(name: str, force: bool = False) -> Path:
    filename, url, desc = MODELS[name]
    dst = MODELS_DIR / filename
    if dst.is_file() and not force:
        return dst
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"baixando {name}: {desc}\n  {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


if __name__ == "__main__":
    for n in sys.argv[1:] or list(MODELS):
        download(n)
