"""Onde rodam as redes neurais pesadas: GPU (CUDA) quando existir, CPU nos outros PCs.

As redes grandes (LaMa, detector de riscos, rosto, DDColor) rodam no ONNX
Runtime, que usa a placa NVIDIA pelo CUDA quando ela está disponível e cai
sozinho para a CPU quando não está. O resultado é o mesmo; só muda o tempo.

Para simular um PC sem placa de vídeo::

    set RETRATO_VIVO_CPU=1        (Windows)
    export RETRATO_VIVO_CPU=1     (Linux/macOS)
"""

from __future__ import annotations

import importlib.util
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
    """A GPU funciona DE FATO? Não basta o CUDA aparecer na lista do ONNX Runtime:
    sem as bibliotecas do CUDA (pacote ``onnxruntime-gpu[cuda,cudnn]``) ele cai para a
    CPU em silêncio. Por isso abre uma sessão de teste e vê onde ela ficou."""
    if ort is None or os.environ.get("RETRATO_VIVO_CPU"):
        return False
    if _GPU not in ort.get_available_providers():
        return False
    # Com o pacote onnxruntime-gpu[cuda,cudnn] as DLLs do CUDA vêm pelo pip; isto as
    # carrega antes da primeira sessão. Só tenta se esse pacote estiver instalado, para
    # não encher o terminal de erros num PC sem placa NVIDIA.
    if hasattr(ort, "preload_dlls") and importlib.util.find_spec("nvidia") is not None:
        try:
            ort.preload_dlls()
        except Exception:
            pass
    probe = _probe_model()
    if probe is None:            # nenhum modelo baixado: nada vai rodar mesmo
        return False
    options = ort.SessionOptions()
    options.log_severity_level = 4
    ort.set_default_logger_severity(4)      # silencioso: sem CUDA a falha aqui é esperada
    try:
        sess = ort.InferenceSession(str(probe), options, providers=[_GPU, _CPU])
    except Exception:
        return False
    finally:
        ort.set_default_logger_severity(3)  # depois, só erros de verdade
    return _GPU in sess.get_providers()


def _probe_model() -> Path | None:
    """O menor modelo ONNX baixado (o YuNet tem 0,2 MB), para a sessão de teste."""
    from .models import MODELS_DIR
    candidates = sorted(MODELS_DIR.glob("*.onnx"), key=lambda p: p.stat().st_size)
    return candidates[0] if candidates else None


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
