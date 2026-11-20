"""Navegação pelas fotos da pasta: anterior, próxima e lista para pular direto.

A pasta é a da foto aberta. A lista se atualiza sozinha quando arquivos são
colocados ou tirados da pasta (``QFileSystemWatcher``), então dá para ir
jogando fotos novas em ``imagens/`` com a janela aberta.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QObject, Signal
from PySide6.QtWidgets import QComboBox

from ..io_utils import list_images

DEFAULT_FOLDER = Path("imagens")    # pasta aberta ao iniciar a interface sem argumentos


def first_photo(folder: str | Path = DEFAULT_FOLDER) -> Path | None:
    """A primeira foto (em ordem alfabética) da pasta, ou ``None`` se não houver."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    photos = list_images(folder)
    return photos[0] if photos else None


class PhotoFolder(QObject):
    """As fotos da pasta da foto aberta, em ordem alfabética."""

    changed = Signal()      # a lista de fotos mudou (pasta nova, arquivo novo ou removido)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.folder: Path | None = None
        self.photos: list[Path] = []
        self.watcher = QFileSystemWatcher(self)
        self.watcher.directoryChanged.connect(self.refresh)

    def follow(self, photo: Path) -> None:
        """Passa a acompanhar a pasta de ``photo``."""
        folder = photo.resolve().parent
        if folder != self.folder:
            if self.watcher.directories():
                self.watcher.removePaths(self.watcher.directories())
            self.folder = folder
            self.watcher.addPath(str(folder))
        self.refresh()

    def refresh(self) -> None:
        if self.folder is None or not self.folder.is_dir():
            self.photos = []
        else:
            self.photos = [p.resolve() for p in list_images(self.folder)]
        self.changed.emit()

    def index_of(self, photo: Path | None) -> int:
        """Posição de ``photo`` na lista, ou -1 se ela não estiver lá."""
        if photo is None:
            return -1
        try:
            return self.photos.index(photo.resolve())
        except ValueError:
            return -1

    def neighbor(self, photo: Path | None, step: int) -> Path | None:
        """A foto ``step`` posições depois de ``photo`` (negativo = antes), dando a volta
        no fim da lista. Se ``photo`` saiu da pasta, começa do início (ou do fim)."""
        if not self.photos:
            return None
        i = self.index_of(photo)
        if i < 0:
            return self.photos[0 if step > 0 else -1]
        return self.photos[(i + step) % len(self.photos)]


class PhotoPicker(QComboBox):
    """Lista com os nomes das fotos da pasta. Escolher um nome abre a foto."""

    def __init__(self, folder: PhotoFolder) -> None:
        super().__init__()
        self.folder = folder
        self.setToolTip("Fotos da pasta da foto aberta (atualiza sozinha)")
        self.setMinimumContentsLength(18)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

    def show_photos(self, current: Path | None) -> None:
        """Recarrega os nomes e marca ``current``, sem disparar a abertura de foto."""
        self.blockSignals(True)
        self.clear()
        for photo in self.folder.photos:
            self.addItem(photo.name, str(photo))
        self.setCurrentIndex(self.folder.index_of(current))
        self.blockSignals(False)
        self.setEnabled(bool(self.folder.photos))
