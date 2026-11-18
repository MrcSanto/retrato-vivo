"""Roda a restauração fora da thread da interface, para a janela não travar."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..pipeline import RestorationConfig, restore


class RestoreWorker(QThread):
    """Chama ``restore()`` em segundo plano e avisa a janela por um sinal quando termina."""

    done = Signal(object)     # o RestorationResult
    failed = Signal(str)      # a mensagem do erro

    def __init__(self, image: np.ndarray, config: RestorationConfig, **restore_options) -> None:
        """``restore_options`` são os outros argumentos nomeados de ``restore()``, como ``post_stages``."""
        super().__init__()
        self._image, self._config, self._restore_options = image, config, restore_options

    def run(self) -> None:
        """O Qt executa este método na thread nova quando alguém chama ``start()``."""
        try:
            result = restore(self._image, self._config, **self._restore_options)
        # Pega qualquer erro de propósito: uma exceção solta aqui mataria a thread
        # em silêncio e a janela ficaria esperando para sempre.
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.done.emit(result)
