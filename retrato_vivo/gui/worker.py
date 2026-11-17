"""Roda a restauração fora da thread da interface, para a janela não travar."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..pipeline import restore


class RestoreWorker(QThread):
    """Chama ``restore()`` em segundo plano e avisa a janela por um sinal quando termina."""

    done = Signal(object)     # o RestorationResult
    failed = Signal(str)      # a mensagem do erro

    def __init__(self, image: np.ndarray) -> None:
        super().__init__()
        self._image = image

    def run(self) -> None:
        """O Qt executa este método na thread nova quando alguém chama ``start()``."""
        try:
            result = restore(self._image)
        # Pega qualquer erro de propósito: uma exceção solta aqui mataria a thread
        # em silêncio e a janela ficaria esperando para sempre.
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.done.emit(result)
