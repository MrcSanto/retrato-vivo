"""Converte o detector de riscos da Microsoft (PyTorch) para ONNX.

Rede: U-Net de detecção de riscos do "Bringing Old Photos Back to Life"
(Wan et al., CVPR 2020), licença MIT. Treinada pelos autores em fotos antigas
reais anotadas. Este script só CONVERTE os pesos oficiais; não treina nada.

Precisa do PyTorch (não faz parte do requirements.txt: só quem gera o modelo
precisa dele). Os outros membros da equipe só baixam o .onnx pronto com
``python -m retrato_vivo modelos detector_riscos``.

Uso::

    pip install torch
    python scripts/exportar_detector_riscos.py           # gera models/scratch_detector.onnx
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "models" / "scratch_detector.onnx"

CODE_URL = ("https://raw.githubusercontent.com/microsoft/Bringing-Old-Photos-Back-to-Life/"
            "master/Global/detection_models/")
WEIGHTS_URL = ("https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/"
               "resolve/main/Global/checkpoints/detection/FT_Epoch_latest.pt")
WEIGHTS_SHA256 = "b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dst: Path) -> Path:
    if not dst.is_file():
        print(f"baixando {url}")
        urllib.request.urlretrieve(url, dst)
    return dst


def build_unet(work: Path):
    """Recria a U-Net com o código oficial (networks.py + antialiasing.py)."""
    pkg = work / "detection_models"
    pkg.mkdir(exist_ok=True)
    for name in ("networks.py", "antialiasing.py"):
        fetch(CODE_URL + name, pkg / name)
    (pkg / "__init__.py").touch()
    # networks.py importa um BatchNorm sincronizado usado só no treino com várias GPUs.
    sys.modules["detection_models.sync_batchnorm"] = types.SimpleNamespace(
        DataParallelWithCallback=lambda m: m)
    sys.path.insert(0, str(work))
    from detection_models import networks
    return networks.UNet(in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6,
                         padding=True, batch_norm=True, up_mode="upsample",
                         with_tanh=False, sync_bn=False, antialiasing=True)



def main() -> int:
    # O exportador do PyTorch imprime emojis, que o terminal do Windows (cp1252) não aceita.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    import numpy as np
    import torch

    work =Path(tempfile.gettempdir()) / "retrato_vivo_exportar_riscos"
    work.mkdir(exist_ok=True)
    weights = fetch(WEIGHTS_URL, work / "FT_Epoch_latest.pt")
    if sha256(weights) != WEIGHTS_SHA256:
        print("ERRO: os pesos baixados não conferem com o hash esperado.")
        return 1

    unet = build_unet(work)
    # weights_only=True: carrega só tensores, sem executar código do arquivo.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    unet.load_state_dict(checkpoint["model_state"])
    model = torch.nn.Sequential(unet, torch.nn.Sigmoid()).eval()

    # Entrada: foto em cinza normalizada para [-1, 1], altura e largura múltiplas de 16.
    dummy = torch.zeros(1, 1, 256, 384)
    OUTPUT.parent.mkdir(exist_ok=True)
    torch.onnx.export(model, dummy, str(OUTPUT), input_names=["gray"], output_names=["scratch_prob"],
                      dynamic_axes={"gray": {2: "h", 3: "w"}, "scratch_prob": {2: "h", 3: "w"}},
                      opset_version=17)
    # O exportador novo grava os pesos num arquivo .data separado. Junta tudo num
    # .onnx só (150 MB, bem abaixo do limite de 2 GB): é mais fácil de distribuir.
    import onnx
    proto = onnx.load(str(OUTPUT), load_external_data=True)
    onnx.save(proto, str(OUTPUT), save_as_external_data=False)
    Path(str(OUTPUT) + ".data").unlink(missing_ok=True)

    # Confere: o ONNX tem de dar o mesmo resultado que o PyTorch, em outro tamanho.
    x = torch.rand(1, 1, 320, 480) * 2 - 1
    with torch.no_grad():
        ref = model(x).numpy()
    try:
        import onnxruntime as ort
        got = ort.InferenceSession(str(OUTPUT), providers=["CPUExecutionProvider"]).run(
            None, {"gray": x.numpy()})[0]
    except ImportError:                     # sem onnxruntime neste Python: confere pelo OpenCV
        import cv2
        net = cv2.dnn.readNetFromONNX(str(OUTPUT))
        net.setInput(x.numpy())
        got = net.forward()
    diff = float(np.abs(ref - got).max())
    print(f"gerado {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB); diferença máxima para o PyTorch: {diff:.2e}")
    return 0 if diff < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
