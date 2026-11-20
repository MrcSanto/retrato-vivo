"""Onde rodam as redes neurais pesadas: GPU (CUDA) quando existir, CPU nos outros PCs.

As redes grandes (LaMa, detector de riscos, rosto, DDColor) rodam no ONNX
Runtime, que usa a placa NVIDIA pelo CUDA quando ela está disponível e cai
sozinho para a CPU quando não está. O resultado é o mesmo; só muda o tempo.

Para simular um PC sem placa de vídeo::

    set RETRATO_VIVO_CPU=1        (Windows)
    export RETRATO_VIVO_CPU=1     (Linux/macOS)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:          # sem onnxruntime, as etapas de IA pesada ficam indisponíveis
    ort = None

_GPU = "CUDAExecutionProvider"
_CPU = "CPUExecutionProvider"


def is_available() -> bool:
    """True se o ONNX Runtime está instalado."""
    return ort is not None


@lru_cache(maxsize=1)
def _gpu_usable() -> bool:
    if ort is None or os.environ.get("RETRATO_VIVO_CPU"):
        return False
    if _GPU not in ort.get_available_providers():
        return False
    # Com o pacote onnxruntime-gpu[cuda,cudnn] as DLLs do CUDA vêm pelo pip;
    # isto as carrega antes da primeira sessão. Sem placa NVIDIA, falha em silêncio.
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            return False
    return True


def device_name() -> str:
    """Texto para a interface: "GPU (CUDA)" ou "CPU"."""
    return "GPU (CUDA)" if _gpu_usable() else "CPU"


@lru_cache(maxsize=None)
def session(model_path: str | Path):
    """Sessão do ONNX Runtime para o modelo (criada uma vez e reaproveitada).

    Tenta a GPU primeiro. Se a sessão na GPU falhar (driver antigo, memória
    cheia), cria na CPU: a ferramenta precisa funcionar em qualquer PC.
    """
    if ort is None:
        raise RuntimeError("onnxruntime não está instalado (pip install -r requirements.txt)")
    options = ort.SessionOptions()
    options.log_severity_level = 3          # só erros; os avisos de otimização poluem o terminal
    path = str(model_path)
    if _gpu_usable():
        try:
            return ort.InferenceSession(path, options, providers=[_GPU, _CPU])
        except Exception:
            pass
    return ort.InferenceSession(path, options, providers=[_CPU])


def session_device(sess) -> str:
    """Em que dispositivo uma sessão está rodando de fato."""
    return "GPU (CUDA)" if _GPU in sess.get_providers() else "CPU"
