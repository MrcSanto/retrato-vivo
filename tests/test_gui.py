"""Testes da interface gráfica. Rodar com ``python -m pytest``.

Rodam sem tela: a plataforma "offscreen" do Qt desenha as janelas na memória.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPointF, QRect, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDropEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from retrato_vivo.gui.convert import fit_rect, to_qimage
from retrato_vivo.gui.main_window import MainWindow
from retrato_vivo.gui.photo_view import PhotoView
from retrato_vivo.io_utils import save_image

RED_BGR, BLUE_BGR = (0, 0, 255), (255, 0, 0)     # o OpenCV guarda os canais como azul, verde, vermelho
RED, BLUE = QColor(255, 0, 0), QColor(0, 0, 255)


@pytest.fixture(scope="module")
def app():
    """Qt exige uma única QApplication viva antes de criar qualquer widget."""
    return QApplication.instance() or QApplication([])


def two_pixel_photo() -> np.ndarray:
    """Foto 3x2 preta, com o canto superior esquerdo vermelho e o inferior direito azul."""
    img = np.zeros((2, 3, 3), np.uint8)
    img[0, 0] = RED_BGR
    img[1, 2] = BLUE_BGR
    return img


def red_and_blue_photo() -> np.ndarray:
    """Foto 200x100: metade esquerda vermelha, metade direita azul."""
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = RED_BGR
    img[:, 100:] = BLUE_BGR
    return img


def file_mime(path) -> QMimeData:
    """O que o sistema entrega quando o usuário arrasta um arquivo para a janela."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    return mime


@pytest.fixture
def photo_file(tmp_path):
    return save_image(tmp_path / "vovó.png", red_and_blue_photo())


@pytest.fixture
def warnings(monkeypatch):
    """Troca a caixa de aviso (que travaria o teste esperando um clique) por uma lista de textos."""
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append(text))
    return shown


# ================================================================ convert.py

def test_to_qimage_keeps_size_and_colors():
    qimage = to_qimage(two_pixel_photo())

    assert (qimage.width(), qimage.height()) == (3, 2)
    assert qimage.pixelColor(0, 0) == RED
    assert qimage.pixelColor(2, 1) == BLUE


def test_to_qimage_does_not_depend_on_the_array_afterwards():
    img = two_pixel_photo()
    qimage = to_qimage(img)

    img[:] = 0

    assert qimage.pixelColor(0, 0) == RED


def test_to_qimage_accepts_a_slice_of_another_image():
    wide = np.zeros((2, 6, 3), np.uint8)
    wide[1, 4] = BLUE_BGR

    qimage = to_qimage(wide[:, ::2])      # colunas 0, 2 e 4: a memória não é contínua

    assert (qimage.width(), qimage.height()) == (3, 2)
    assert qimage.pixelColor(2, 1) == BLUE


@pytest.mark.parametrize("image_size, area, expected", [
    (QSize(400, 200), QRect(0, 0, 100, 100), QRect(0, 25, 100, 50)),       # foto deitada: sobra em cima e embaixo
    (QSize(200, 400), QRect(10, 20, 300, 100), QRect(135, 20, 50, 100)),   # foto em pé: sobra nos lados
])
def test_fit_rect_keeps_proportion_and_centers(image_size, area, expected):
    assert fit_rect(image_size, area) == expected


# ============================================================= photo_view.py

def test_photo_view_draws_the_photo_centered_keeping_proportion(app):
    view = PhotoView()
    view.resize(200, 200)

    view.set_photo(red_and_blue_photo())       # 200x100 numa área 200x200: ocupa a faixa do meio

    shot = view.grab().toImage()
    assert shot.pixelColor(50, 100) == RED
    assert shot.pixelColor(150, 100) == BLUE
    assert shot.pixelColor(50, 10) not in (RED, BLUE)      # acima da foto: só o fundo


# ============================================================ main_window.py

def test_new_window_shows_a_hint_in_the_status_bar(app):
    window = MainWindow()

    assert window.statusBar().currentMessage() != ""


def test_opening_a_photo_shows_it_and_names_it_in_the_status_bar(app, photo_file):
    window = MainWindow()
    window.show()

    window.open_photo(photo_file)

    assert np.array_equal(window.image, red_and_blue_photo())
    assert "vovó.png" in window.statusBar().currentMessage()
    view = window.tabs.currentWidget()
    shot = view.grab().toImage()
    assert shot.pixelColor(view.width() // 4, view.height() // 2) == RED
    assert shot.pixelColor(3 * view.width() // 4, view.height() // 2) == BLUE


def test_opening_a_broken_file_warns_and_keeps_the_current_photo(app, photo_file, tmp_path, warnings):
    broken = tmp_path / "quebrada.jpg"
    broken.write_bytes(b"isto nao e uma imagem")
    window = MainWindow()
    window.open_photo(photo_file)

    window.open_photo(broken)

    assert len(warnings) == 1 and "quebrada.jpg" in warnings[0]
    assert np.array_equal(window.image, red_and_blue_photo())
    assert "vovó.png" in window.statusBar().currentMessage()


def test_choosing_a_photo_in_the_dialog_opens_it(app, photo_file, monkeypatch):
    offered_filters = []

    def fake_dialog(parent, title, folder, filters):
        offered_filters.append(filters)
        return str(photo_file), ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_dialog)
    window = MainWindow()

    window.choose_photo()

    assert np.array_equal(window.image, red_and_blue_photo())
    assert "*.webp" in offered_filters[0] and "*.tiff" in offered_filters[0]


def test_cancelling_the_dialog_changes_nothing(app, monkeypatch, warnings):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    window = MainWindow()

    window.choose_photo()

    assert window.image is None
    assert warnings == []


def test_dropping_a_photo_file_opens_it(app, photo_file):
    window = MainWindow()
    mime = file_mime(photo_file)
    drop = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime,
                      Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)

    window.dropEvent(drop)

    assert np.array_equal(window.image, red_and_blue_photo())


class FakeDrag:
    """O mínimo de um ``QDragEnterEvent``: o que está sendo arrastado e se a janela aceitou."""

    def __init__(self, mime: QMimeData) -> None:
        self.mime, self.accepted = mime, False

    def mimeData(self) -> QMimeData:
        return self.mime

    def acceptProposedAction(self) -> None:
        self.accepted = True


def test_drag_is_accepted_for_files_and_refused_for_text(app, photo_file):
    window = MainWindow()
    text_mime = QMimeData()
    text_mime.setText("um texto qualquer")
    file_drag, text_drag = FakeDrag(file_mime(photo_file)), FakeDrag(text_mime)

    window.dragEnterEvent(file_drag)
    window.dragEnterEvent(text_drag)

    assert file_drag.accepted
    assert not text_drag.accepted
