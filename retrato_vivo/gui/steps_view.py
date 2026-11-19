"""Aba "Etapas": a foto depois de cada etapa do pipeline, com a explicação do que foi feito."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QVBoxLayout, QWidget

from ..pipeline import STEP_INFO, RestorationResult
from .photo_view import PhotoView

LIST_WIDTH = 200   # em pixels da tela


def step_title(name: str) -> str:
    """Nome para mostrar ao usuário. O que não está em ``STEP_INFO`` aparece com o próprio nome."""
    return STEP_INFO[name][0] if name in STEP_INFO else name.replace("_", " ")


def footer_text(result: RestorationResult) -> str:
    """Rodapé da aba: as etapas que o pipeline pulou, com o motivo, e o tempo de cada fase."""
    # Etapa pulada tem nota em result.notes, mas não tem imagem em result.steps.
    skipped = [f"{step_title(name)}: {note}" for name, note in result.notes.items()
               if name not in result.steps]
    timings = " · ".join(f"{step_title(phase)} {ms:.0f} ms" for phase, ms in result.timings_ms.items())
    return "\n".join([*skipped, f"Tempo por fase: {timings}"])


class StepsView(QWidget):
    """Lista as etapas aplicadas. Ao escolher uma, mostra a foto naquele ponto e a explicação."""

    def __init__(self) -> None:
        super().__init__()
        self._result: RestorationResult | None = None
        self._names: list[str] = []      # a etapa de cada linha da lista, na mesma ordem

        self.step_list = QListWidget()
        self.step_list.setFixedWidth(LIST_WIDTH)
        self.photo = PhotoView()
        self.caption = QLabel()          # descrição da etapa e o que ela decidiu nesta foto
        self.caption.setWordWrap(True)
        self.footer = QLabel()
        self.footer.setWordWrap(True)

        photo_column = QVBoxLayout()
        photo_column.addWidget(self.photo, stretch=1)
        photo_column.addWidget(self.caption)
        list_and_photo = QHBoxLayout()
        list_and_photo.addWidget(self.step_list)
        list_and_photo.addLayout(photo_column, stretch=1)
        layout = QVBoxLayout(self)
        layout.addLayout(list_and_photo, stretch=1)
        layout.addWidget(self.footer)

        self.step_list.currentRowChanged.connect(self._show_step)

    def set_result(self, result: RestorationResult) -> None:
        row = self.step_list.currentRow()
        previous = self._names[row] if row >= 0 else None
        self._result, self._names = result, list(result.steps)
        self.step_list.clear()
        self.step_list.addItems([step_title(name) for name in self._names])
        # Refazer a restauração não tira o usuário da etapa que ele estava olhando.
        self.step_list.setCurrentRow(self._names.index(previous) if previous in self._names else 0)
        self.footer.setText(footer_text(result))

    def clear(self) -> None:
        self._result, self._names = None, []
        self.step_list.clear()
        self.photo.clear()
        self.caption.clear()
        self.footer.clear()

    def _show_step(self, row: int) -> None:
        """Chamado pelo Qt quando a seleção da lista muda. ``row`` é -1 quando a lista esvazia."""
        if row < 0:
            return
        name = self._names[row]
        _, description = STEP_INFO.get(name, ("", ""))
        note = self._result.notes.get(name, "")
        self.photo.set_photo(self._result.steps[name])
        self.caption.setText("\n".join(text for text in (description, note) if text))
