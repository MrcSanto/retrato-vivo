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
    # Preenchimento com IA (Igor, inpaint.py). Roda no ONNX Runtime (runtime.py).
    "lama": (
        "lama_fp32.onnx",
        "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        "LaMa (Suvorov et al., WACV 2022, Apache-2.0): preenche rasgos e áreas grandes (208 MB)",
    ),
    # Detector de riscos com IA (Igor, damage_ai.py). Convertido dos pesos oficiais da
    # Microsoft por scripts/exportar_detector_riscos.py.
    "detector_riscos": (
        "scratch_detector.onnx",
        "",   # PENDENTE: publicar o .onnx como anexo de um Release do repositório
        "Detector de riscos (Wan et al., CVPR 2020, Microsoft, MIT) (151 MB)",
    ),
    # Restauração de rosto e colorização DDColor (ai/face_restore.py, ai/ddcolor.py). ONNX Runtime.
    "gfpgan": (
        "gfpgan_1.4.onnx",
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
        "GFPGAN v1.4 (Wang et al., CVPR 2021, Apache-2.0): restaura rostos (340 MB)",
    ),
    "ddcolor": (
        "ddcolor.onnx",
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ddcolor.onnx",
        "DDColor (Kang et al., ICCV 2023, Apache-2.0): colorização realista (980 MB)",
    ),
    # Retrato Vivo (living_portrait.py): profundidade de cada pixel.
    "profundidade": (
        "depth_anything_v2_small.onnx",
        "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
        "Depth Anything V2 Small (Yang et al., NeurIPS 2024, Apache-2.0): profundidade (99 MB)",
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
    if not url:
        print(f"{name}: ainda sem endereço para download ({desc}). Gere com o script em scripts/ "
              f"ou copie {filename} para {MODELS_DIR}.")
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
