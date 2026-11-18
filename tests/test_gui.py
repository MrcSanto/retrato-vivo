"""Testes da interface gráfica. Rodar com ``python -m pytest``.

Rodam sem tela: a plataforma "offscreen" do Qt desenha as janelas na memória.
"""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QMimeData, QPointF, QRect, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDropEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from retrato_vivo.gui import worker as worker_module
from retrato_vivo.gui.compare_view import CompareView
from retrato_vivo.gui.convert import fit_rect, to_qimage
from retrato_vivo.gui.main_window import MainWindow
from retrato_vivo.gui.photo_view import PhotoView
from retrato_vivo.gui.worker import RestoreWorker
from retrato_vivo.io_utils import save_image

RED_BGR, GREEN_BGR, BLUE_BGR = (0, 0, 255), (0, 255, 0), (255, 0, 0)     # o OpenCV guarda os canais como azul, verde, vermelho
RED, GREEN, BLUE = QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255)


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


def solid_photo(bgr: tuple[int, int, int]) -> np.ndarray:
    """Foto 200x100 de uma cor só."""
    return np.full((100, 200, 3), bgr, np.uint8)


def mouse(kind: QEvent.Type, x: float, y: float) -> QMouseEvent:
    """Um clique ou arrasto com o botão esquerdo na posição (x, y) do widget."""
    return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), Qt.MouseButton.LeftButton,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)


def file_mime(path) -> QMimeData:
    """O que o sistema entrega quando o usuário arrasta um arquivo para a janela."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    return mime


def wait_until(condition, timeout_s: float = 5.0) -> None:
    """Deixa o Qt entregar os sinais pendentes até ``condition()`` valer."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "a condição não se cumpriu a tempo"
        QApplication.processEvents()
        time.sleep(0.005)


@pytest.fixture
def window(app):
    window = MainWindow()
    yield window
    window.close()      # espera a restauração em andamento antes de a janela ser destruída


@pytest.fixture
def photo_file(tmp_path):
    return save_image(tmp_path / "vovó.png", red_and_blue_photo())


@pytest.fixture
def broken_restore(monkeypatch):
    """Faz a restauração falhar, como acontece quando falta o arquivo de um modelo."""
    def failing_restore(image, *args, **kwargs):
        raise RuntimeError("modelo ausente")

    monkeypatch.setattr(worker_module, "restore", failing_restore)


@pytest.fixture(autouse=True)
def warnings(monkeypatch):
    """Troca a caixa de aviso por uma lista com os textos mostrados.

    Vale para todos os testes: a caixa de verdade fica esperando um clique, e um
    aviso inesperado deixaria o teste parado para sempre em vez de falhar.
    """
    shown = []

    def record(parent, title, text):
        print(f"aviso mostrado: {text}")      # o pytest exibe isto quando o teste falha
        shown.append(text)

    monkeypatch.setattr(QMessageBox, "warning", record)
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


# =========================================================== compare_view.py

@pytest.fixture
def red_before_blue_after(app) -> CompareView:
    """Comparador 200x100 com o "antes" todo vermelho e o "depois" todo azul. O divisor começa no meio."""
    view = CompareView()
    view.resize(200, 100)
    view.set_comparison(solid_photo(RED_BGR), solid_photo(BLUE_BGR))
    return view


def test_compare_view_shows_before_on_the_left_and_after_on_the_right(red_before_blue_after):
    shot = red_before_blue_after.grab().toImage()

    assert shot.pixelColor(50, 50) == RED
    assert shot.pixelColor(150, 50) == BLUE


def test_clicking_and_dragging_moves_the_divider(red_before_blue_after):
    view = red_before_blue_after

    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, 40, 50))
    after_click = view.grab().toImage()
    view.mouseMoveEvent(mouse(QEvent.Type.MouseMove, 160, 50))
    after_drag = view.grab().toImage()

    assert after_click.pixelColor(20, 50) == RED and after_click.pixelColor(60, 50) == BLUE
    assert after_drag.pixelColor(140, 50) == RED and after_drag.pixelColor(180, 50) == BLUE


def test_divider_stops_at_the_photo_edges(red_before_blue_after):
    view = red_before_blue_after

    view.mouseMoveEvent(mouse(QEvent.Type.MouseMove, 500, 50))      # muito além da borda direita
    split_at_right, all_before = view.split, view.grab().toImage()
    view.mouseMoveEvent(mouse(QEvent.Type.MouseMove, -500, 50))     # muito além da borda esquerda
    split_at_left, all_after = view.split, view.grab().toImage()

    assert (split_at_right, split_at_left) == (1.0, 0.0)            # o divisor não sai de cima da foto
    assert all_before.pixelColor(10, 50) == RED and all_before.pixelColor(190, 50) == RED
    assert all_after.pixelColor(10, 50) == BLUE and all_after.pixelColor(190, 50) == BLUE


def test_divider_follows_the_mouse_when_the_photo_has_side_margins(red_before_blue_after):
    view = red_before_blue_after
    view.resize(400, 100)       # a foto 200x100 fica centralizada, de x=100 a x=300

    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, 150, 50))

    shot = view.grab().toImage()
    assert shot.pixelColor(120, 50) == RED
    assert shot.pixelColor(200, 50) == BLUE
    assert shot.pixelColor(50, 50) not in (RED, BLUE)       # margem: só o fundo


def test_showing_a_single_photo_ends_the_comparison(red_before_blue_after):
    view = red_before_blue_after

    view.set_photo(solid_photo(GREEN_BGR))

    shot = view.grab().toImage()
    assert shot.pixelColor(50, 50) == GREEN
    assert shot.pixelColor(150, 50) == GREEN


# ================================================================= worker.py

def test_worker_emits_the_restored_photo(app):
    photo = red_and_blue_photo()
    results = []
    worker = RestoreWorker(photo)
    worker.done.connect(results.append)

    worker.run()        # chamado direto, sem thread: o sinal chega na mesma hora

    assert len(results) == 1
    assert np.array_equal(results[0].original, photo)
    assert results[0].image.shape == photo.shape


def test_worker_reports_the_error_instead_of_dying(app, broken_restore):
    results, errors = [], []
    worker = RestoreWorker(red_and_blue_photo())
    worker.done.connect(results.append)
    worker.failed.connect(errors.append)

    worker.run()

    assert results == []
    assert len(errors) == 1 and "modelo ausente" in errors[0]


# ============================================================ main_window.py

def test_new_window_shows_a_hint_in_the_status_bar(window):
    assert window.statusBar().currentMessage() != ""


def test_opening_a_photo_shows_it_and_names_it_in_the_status_bar(window, photo_file):
    window.show()

    window.open_photo(photo_file)

    assert np.array_equal(window.image, red_and_blue_photo())
    assert "vovó.png" in window.statusBar().currentMessage()
    view = window.tabs.currentWidget()
    shot = view.grab().toImage()
    assert shot.pixelColor(view.width() // 4, view.height() // 2) == RED
    assert shot.pixelColor(3 * view.width() // 4, view.height() // 2) == BLUE


def test_opening_a_broken_file_warns_and_keeps_the_current_photo(window, photo_file, tmp_path, warnings):
    broken = tmp_path / "quebrada.jpg"
    broken.write_bytes(b"isto nao e uma imagem")
    window.open_photo(photo_file)

    window.open_photo(broken)

    assert len(warnings) == 1 and "quebrada.jpg" in warnings[0]
    assert np.array_equal(window.image, red_and_blue_photo())
    assert "vovó.png" in window.statusBar().currentMessage()


def test_choosing_a_photo_in_the_dialog_opens_it(window, photo_file, monkeypatch):
    offered_filters = []

    def fake_dialog(parent, title, folder, filters):
        offered_filters.append(filters)
        return str(photo_file), ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_dialog)

    window.choose_photo()

    assert np.array_equal(window.image, red_and_blue_photo())
    assert "*.webp" in offered_filters[0] and "*.tiff" in offered_filters[0]


def test_cancelling_the_dialog_changes_nothing(window, monkeypatch, warnings):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))

    window.choose_photo()

    assert window.image is None
    assert warnings == []


def test_dropping_a_photo_file_opens_it(window, photo_file):
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


def test_drag_is_accepted_for_files_and_refused_for_text(window, photo_file):
    text_mime = QMimeData()
    text_mime.setText("um texto qualquer")
    file_drag, text_drag = FakeDrag(file_mime(photo_file)), FakeDrag(text_mime)

    window.dragEnterEvent(file_drag)
    window.dragEnterEvent(text_drag)

    assert file_drag.accepted
    assert not text_drag.accepted


def test_opening_a_photo_restores_it_in_the_background(window, photo_file):
    window.open_photo(photo_file)

    assert window.result is None        # open_photo voltou antes de a restauração terminar
    wait_until(lambda: window.result is not None)
    assert window.result.image.shape == red_and_blue_photo().shape


def test_opening_another_photo_discards_the_previous_result(window, photo_file, tmp_path):
    green_file = save_image(tmp_path / "verde.png", solid_photo(GREEN_BGR))
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)

    window.open_photo(green_file)

    assert window.result is None        # o resultado antigo era de outra foto
    wait_until(lambda: window.result is not None)
    assert np.array_equal(window.result.original, solid_photo(GREEN_BGR))


def test_after_restoring_the_screen_compares_original_and_restored(window, photo_file):
    window.show()
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)

    view = window.tabs.currentWidget()
    shot = view.grab().toImage()
    left = shot.pixelColor(view.width() // 4, view.height() // 2)
    right = shot.pixelColor(3 * view.width() // 4, view.height() // 2)

    blue, green, red = (int(channel) for channel in window.result.image[50, 150])
    assert left == RED                              # a original, intacta
    assert right == QColor(red, green, blue)        # a restaurada
    assert right != BLUE                            # o pipeline altera as cores desta foto


def test_opening_another_photo_ends_the_previous_comparison(window, photo_file, tmp_path):
    green_file = save_image(tmp_path / "verde.png", solid_photo(GREEN_BGR))
    window.show()
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)

    window.open_photo(green_file)

    view = window.tabs.currentWidget()
    shot = view.grab().toImage()
    assert shot.pixelColor(view.width() // 4, view.height() // 2) == GREEN     # nada da foto anterior


def test_window_is_busy_while_restoring_and_free_afterwards(window, photo_file):
    window.open_photo(photo_file)

    assert not window.toolbar.isEnabled()
    assert not window.acceptDrops()
    assert not window.progress.isHidden()

    wait_until(lambda: window.result is not None)
    assert window.toolbar.isEnabled()
    assert window.acceptDrops()
    assert window.progress.isHidden()


def test_restore_button_only_works_with_a_photo_open(window, photo_file):
    assert not window.restore_action.isEnabled()

    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)

    assert window.restore_action.isEnabled()


def test_restore_button_restores_the_photo_again(window, photo_file):
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)
    first = window.result

    window.restore_action.trigger()
    wait_until(lambda: window.result is not first)

    assert window.result.image.shape == first.image.shape


def test_failed_restoration_warns_and_frees_the_window(window, photo_file, warnings, broken_restore):
    window.open_photo(photo_file)
    wait_until(lambda: window.toolbar.isEnabled())

    assert len(warnings) == 1 and "modelo ausente" in warnings[0]
    assert window.result is None


def test_closing_the_window_waits_for_the_running_restoration(window, photo_file):
    window.open_photo(photo_file)

    window.close()
    QApplication.processEvents()        # só entrega os sinais já emitidos; não espera a thread

    assert window.result is not None
