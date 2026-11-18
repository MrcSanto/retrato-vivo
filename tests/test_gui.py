"""Testes da interface gráfica. Rodar com ``python -m pytest``.

Rodam sem tela: a plataforma "offscreen" do Qt desenha as janelas na memória.
"""

import dataclasses
import os
import sys
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QMimeData, QPointF, QRect, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDropEvent, QImage, QKeySequence, QMouseEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QFileDialog, QLabel, QMessageBox

from retrato_vivo import models
# O pacote retrato_vivo.ai exporta a função "colorize", que esconde o módulo de mesmo nome.
from retrato_vivo.ai.colorize import DEFAULT_STRENGTH as COLORIZE_DEFAULT_STRENGTH
from retrato_vivo.ai.colorize import is_available as colorize_model_is_downloaded
from retrato_vivo.gui import controls as controls_module
from retrato_vivo.gui import worker as worker_module
from retrato_vivo.gui.compare_view import CompareView
from retrato_vivo.gui.controls import ControlsPanel
from retrato_vivo.gui.convert import fit_rect, to_qimage
from retrato_vivo.gui.main_window import MainWindow, restoration_summary
from retrato_vivo.gui.mask_editor import MaskCanvas, MaskEditor, overlay_image
from retrato_vivo.gui.photo_view import PhotoView
from retrato_vivo.gui.steps_view import StepsView
from retrato_vivo.gui.worker import RestoreWorker
from retrato_vivo.io_utils import load_image, save_image
from retrato_vivo.pipeline import STEP_INFO, RestorationConfig, restore
from retrato_vivo.visualize import side_by_side

RED_BGR, GREEN_BGR, BLUE_BGR = (0, 0, 255), (0, 255, 0), (255, 0, 0)     # o OpenCV guarda os canais como azul, verde, vermelho
WHITE_BGR = (255, 255, 255)
RED, GREEN, BLUE = QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255)
RED_RGB, GREEN_RGB, BLUE_RGB = (255, 0, 0), (0, 255, 0), (0, 0, 255)     # para comparar com capturas de tela


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


def longest_freeze_until(condition, timeout_s: float = 5.0) -> float:
    """Espera como ``wait_until`` e devolve, em segundos, o maior tempo que o Qt ficou sem devolver o controle.

    Se algum método da janela ficar parado esperando a restauração terminar, esse tempo aparece aqui.
    """
    deadline, longest = time.monotonic() + timeout_s, 0.0
    while not condition():
        assert time.monotonic() < deadline, "a condição não se cumpriu a tempo"
        started = time.monotonic()
        QApplication.processEvents()
        longest = max(longest, time.monotonic() - started)
        time.sleep(0.005)
    return longest


def let_qt_run(seconds: float) -> None:
    """Deixa o Qt trabalhar por um tempo, para o que estiver agendado (timers, sinais) acontecer."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
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
def errors_inside_qt_calls(monkeypatch):
    """Faz o teste falhar quando um método chamado pelo Qt (botão, sinal, evento) levanta exceção.

    O Qt só imprime o erro e segue em frente; sem isto o teste passaria por engano.
    """
    raised = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, traceback: raised.append(error))
    yield
    assert not raised, f"exceção dentro de um método chamado pelo Qt: {raised!r}"


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


def screenshot(widget) -> np.ndarray:
    """O que o widget mostra, como array (altura, largura, 3) em RGB."""
    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return rows[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()


def distinct_colors(widget) -> int:
    """Quantas cores diferentes aparecem no widget. Uma só quer dizer que ele está em branco."""
    return len(np.unique(screenshot(widget).reshape(-1, 3), axis=0))


def pixels_of_another_color(region: np.ndarray, expected_rgb: tuple[int, int, int]) -> int:
    """Quantos pixels da região não têm a cor esperada."""
    return int((region != expected_rgb).any(axis=-1).sum())


def test_empty_photo_view_shows_its_placeholder_text_until_a_photo_arrives(app):
    blank, with_text = PhotoView(), PhotoView()
    for view in (blank, with_text):
        view.resize(300, 100)
    with_text.placeholder = "Arraste uma foto para cá"

    colors_when_blank = distinct_colors(blank)
    colors_with_text = distinct_colors(with_text)
    with_text.set_photo(solid_photo(RED_BGR))
    blank.set_photo(solid_photo(RED_BGR))

    assert colors_when_blank == 1
    assert colors_with_text > 1                                     # o texto foi desenhado
    assert with_text.grab().toImage() == blank.grab().toImage()     # com foto, o texto some


def test_photo_view_can_be_emptied(app):
    view = PhotoView()
    view.resize(200, 100)
    view.set_photo(solid_photo(RED_BGR))

    view.clear()

    assert view.grab().toImage().pixelColor(100, 50) != RED


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


def test_compare_view_labels_each_side_at_the_top(red_before_blue_after):
    shot = screenshot(red_before_blue_after)        # 200x100: "antes" vermelho à esquerda, "depois" azul à direita

    assert pixels_of_another_color(shot[:45, :95], RED_RGB) > 0         # rótulo "Antes"
    assert pixels_of_another_color(shot[:45, 105:], BLUE_RGB) > 0       # rótulo "Depois"
    assert pixels_of_another_color(shot[55:, :95], RED_RGB) == 0        # o resto da foto fica livre
    assert pixels_of_another_color(shot[55:, 105:], BLUE_RGB) == 0


def test_label_disappears_when_its_side_gets_too_narrow(red_before_blue_after):
    view = red_before_blue_after

    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, 20, 50))      # sobra uma faixa de 20 pixels de "antes"

    shot = screenshot(view)
    assert pixels_of_another_color(shot[:45, :17], RED_RGB) == 0        # sem rótulo "Antes" na faixa estreita
    assert pixels_of_another_color(shot[:45, 24:110], BLUE_RGB) == 0    # e ele não invade o lado do "depois"
    assert pixels_of_another_color(shot[:45, 110:], BLUE_RGB) > 0       # "Depois" continua lá


def test_after_label_also_disappears_when_its_side_gets_too_narrow(red_before_blue_after):
    view = red_before_blue_after

    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, 180, 50))     # sobra uma faixa de 20 pixels de "depois"

    shot = screenshot(view)
    assert pixels_of_another_color(shot[:45, 184:], BLUE_RGB) == 0      # sem rótulo "Depois" na faixa estreita
    assert pixels_of_another_color(shot[:45, 90:176], RED_RGB) == 0     # e ele não invade o lado do "antes"
    assert pixels_of_another_color(shot[:45, :90], RED_RGB) > 0         # "Antes" continua lá


def test_divider_shows_even_over_a_white_photo(app):
    view = CompareView()
    view.resize(200, 100)
    view.set_comparison(solid_photo(WHITE_BGR), solid_photo(WHITE_BGR))

    row = screenshot(view)[70]          # uma linha da foto abaixo dos rótulos

    assert row[95:105].min() < 100                                      # há um traço escuro junto ao divisor
    assert (row[:95] == 255).all() and (row[105:] == 255).all()         # e só ali


def test_single_photo_has_no_labels_and_no_divider(red_before_blue_after):
    red_before_blue_after.set_photo(solid_photo(GREEN_BGR))

    assert pixels_of_another_color(screenshot(red_before_blue_after), GREEN_RGB) == 0


def test_showing_a_single_photo_ends_the_comparison(red_before_blue_after):
    view = red_before_blue_after

    view.set_photo(solid_photo(GREEN_BGR))

    shot = view.grab().toImage()
    assert shot.pixelColor(50, 50) == GREEN
    assert shot.pixelColor(150, 50) == GREEN


# ============================================================= steps_view.py

@pytest.fixture
def three_step_result():
    """Resultado de verdade, com as etapas trocadas por fotos de uma cor só, fáceis de reconhecer na tela."""
    result = restore(red_and_blue_photo())
    result.steps = {"original": solid_photo(RED_BGR), "mapa_de_danos": solid_photo(GREEN_BGR),
                    "etapa_nova": solid_photo(BLUE_BGR)}
    result.notes = {"mapa_de_danos": "8 riscos, 5 vincos", "ruido_removido": "pulada: foto limpa"}
    result.timings_ms = {"diagnostico": 57.4, "detectar_danos": 206.0}
    return result


@pytest.fixture
def steps_view(app, three_step_result) -> StepsView:
    view = StepsView()
    view.resize(600, 400)
    view.set_result(three_step_result)
    return view


def photo_color(view: StepsView) -> QColor:
    """A cor no centro da foto que a aba está mostrando."""
    return view.photo.grab().toImage().pixelColor(view.photo.width() // 2, view.photo.height() // 2)


def listed_titles(view: StepsView) -> list[str]:
    return [view.step_list.item(row).text() for row in range(view.step_list.count())]


def test_steps_view_lists_the_applied_steps_and_starts_at_the_first(steps_view):
    known_titles = [STEP_INFO["original"][0], STEP_INFO["mapa_de_danos"][0]]

    assert listed_titles(steps_view) == [*known_titles, "etapa nova"]     # etapa fora de STEP_INFO: o próprio nome
    assert photo_color(steps_view) == RED


def test_selecting_a_step_shows_its_photo_description_and_note(steps_view):
    steps_view.step_list.setCurrentRow(1)

    assert photo_color(steps_view) == GREEN
    assert STEP_INFO["mapa_de_danos"][1] in steps_view.caption.text()
    assert "8 riscos, 5 vincos" in steps_view.caption.text()


def test_step_without_description_or_note_has_an_empty_caption(steps_view):
    steps_view.step_list.setCurrentRow(2)

    assert photo_color(steps_view) == BLUE
    assert steps_view.caption.text() == ""


def test_redoing_the_restoration_keeps_the_selected_step(steps_view, three_step_result):
    steps_view.step_list.setCurrentRow(1)

    steps_view.set_result(three_step_result)

    assert steps_view.step_list.currentRow() == 1
    assert photo_color(steps_view) == GREEN


def test_selection_returns_to_the_first_step_when_the_selected_one_is_gone(steps_view, three_step_result):
    steps_view.step_list.setCurrentRow(2)
    first_two_steps = dict(list(three_step_result.steps.items())[:2])
    result_without_the_selected_step = dataclasses.replace(three_step_result, steps=first_two_steps)

    steps_view.set_result(result_without_the_selected_step)

    assert steps_view.step_list.currentRow() == 0
    assert photo_color(steps_view) == RED


def test_footer_explains_skipped_steps_and_shows_the_time_of_each_phase(steps_view):
    footer = steps_view.footer.text()

    assert f"{STEP_INFO['ruido_removido'][0]}: pulada: foto limpa" in footer
    assert "8 riscos, 5 vincos" not in footer       # nota de etapa aplicada fica na legenda da etapa
    assert "diagnostico 57 ms" in footer
    assert "detectar danos 206 ms" in footer


def test_cleared_steps_view_shows_nothing(steps_view):
    steps_view.clear()

    assert listed_titles(steps_view) == []
    assert steps_view.caption.text() == "" and steps_view.footer.text() == ""
    assert photo_color(steps_view) not in (RED, GREEN, BLUE)


# ============================================================ mask_editor.py

PRESS, MOVE, RELEASE = QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease


@pytest.fixture
def canvas(app) -> MaskCanvas:
    """Área de pintura 200x100 mostrando uma foto vermelha de 400x200: cada pixel da tela vale 2 da foto."""
    canvas = MaskCanvas()
    canvas.resize(200, 100)
    canvas.set_photo(np.full((200, 400, 3), RED_BGR, np.uint8))
    canvas.brush_size = 10      # na tela; na foto o traço fica com 20 pixels de largura
    return canvas


def drag(canvas: MaskCanvas, start: tuple[int, int], end: tuple[int, int]) -> None:
    """Aperta o botão em ``start``, arrasta até ``end`` e solta (posições na tela)."""
    canvas.mousePressEvent(mouse(PRESS, *start))
    canvas.mouseMoveEvent(mouse(MOVE, *end))
    canvas.mouseReleaseEvent(mouse(RELEASE, *end))


def test_new_photo_gets_empty_masks_of_its_size(canvas):
    for mask in (canvas.mask_add, canvas.mask_remove):
        assert mask.shape == (200, 400) and mask.dtype == np.uint8
        assert not mask.any()


def test_brush_stroke_marks_mask_add_under_the_mouse(canvas):
    drag(canvas, (50, 50), (150, 50))       # na foto: da coluna 100 à 300, na linha 100

    assert canvas.mask_add[100, 100] == canvas.mask_add[100, 200] == canvas.mask_add[100, 300] == 255
    assert canvas.mask_add[106, 200] == 255         # dentro da largura do traço
    assert canvas.mask_add[130, 200] == 0           # fora dela
    assert canvas.mask_add[100, 350] == 0           # depois do fim do traço
    assert not canvas.mask_remove.any()


def test_eraser_stroke_marks_mask_remove_instead(canvas):
    canvas.set_erasing(True)

    drag(canvas, (50, 50), (150, 50))

    assert canvas.mask_remove[100, 200] == 255
    assert not canvas.mask_add.any()


def test_each_tool_undoes_the_other_where_they_cross(canvas):
    drag(canvas, (50, 50), (150, 50))                   # pincel
    canvas.set_erasing(True)
    drag(canvas, (100, 20), (100, 80))                  # borracha, cruzando o traço na coluna 200
    crossing_after_eraser = (canvas.mask_add[100, 200], canvas.mask_remove[100, 200])
    canvas.set_erasing(False)
    drag(canvas, (50, 50), (150, 50))                   # pincel de novo, por cima
    crossing_after_brush = (canvas.mask_add[100, 200], canvas.mask_remove[100, 200])

    assert crossing_after_eraser == (0, 255)
    assert crossing_after_brush == (255, 0)
    assert canvas.mask_add[100, 120] == 255             # longe do cruzamento, o primeiro traço continua


def test_click_without_dragging_paints_a_dot_of_the_brush_size(canvas):
    canvas.mousePressEvent(mouse(PRESS, 100, 50))       # na foto: (200, 100); o ponto tem raio 10
    canvas.mouseReleaseEvent(mouse(RELEASE, 100, 50))

    assert canvas.mask_add[100, 200] == 255
    assert canvas.mask_add[100, 206] == 255
    assert canvas.mask_add[100, 216] == 0


def test_bigger_brush_paints_a_wider_stroke(canvas):
    canvas.set_brush_size(30)       # na foto: 60 pixels de largura

    drag(canvas, (50, 50), (150, 50))

    assert canvas.mask_add[125, 200] == 255
    assert canvas.mask_add[140, 200] == 0


def test_moving_the_mouse_after_releasing_paints_nothing(canvas):
    drag(canvas, (50, 50), (60, 50))

    canvas.mouseMoveEvent(mouse(MOVE, 180, 90))

    assert canvas.mask_add[180, 360] == 0


def test_stroke_follows_the_photo_when_it_has_side_margins(canvas):
    canvas.resize(400, 100)         # a foto fica de x=100 a x=300 na tela, com 2 pixels da foto por pixel

    canvas.mousePressEvent(mouse(PRESS, 150, 50))       # 50 pixels de tela depois da borda da foto
    canvas.mouseReleaseEvent(mouse(RELEASE, 150, 50))

    assert canvas.mask_add[100, 100] == 255
    assert canvas.mask_add[100, 300] == 0


def test_clearing_empties_both_masks(canvas):
    drag(canvas, (50, 50), (150, 50))
    canvas.set_erasing(True)
    drag(canvas, (50, 80), (150, 80))

    canvas.clear_masks()

    assert not canvas.mask_add.any() and not canvas.mask_remove.any()


def test_strokes_are_drawn_over_the_photo(canvas):
    drag(canvas, (50, 50), (150, 50))

    shot = canvas.grab().toImage()
    assert shot.pixelColor(100, 50) != RED          # o traço, por cima do vermelho
    assert shot.pixelColor(100, 85) == RED          # fora do traço, a foto intacta


def test_same_size_photo_keeps_the_strokes_and_another_size_resets_them(canvas):
    drag(canvas, (50, 50), (150, 50))

    canvas.set_photo(np.full((200, 400, 3), BLUE_BGR, np.uint8))        # a mesma foto restaurada de novo
    kept = bool(canvas.mask_add.any())
    canvas.set_photo(np.full((100, 300, 3), BLUE_BGR, np.uint8))        # outra foto

    assert kept
    assert canvas.mask_add.shape == canvas.mask_remove.shape == (100, 300)
    assert not canvas.mask_add.any()


def test_overlay_is_transparent_except_where_the_masks_are_painted(app):
    mask_add, mask_remove = np.zeros((2, 3), np.uint8), np.zeros((2, 3), np.uint8)
    mask_add[0, 0] = 255
    mask_remove[1, 2] = 255

    overlay = overlay_image(mask_add, mask_remove)

    assert (overlay.width(), overlay.height()) == (3, 2)
    assert overlay.pixelColor(1, 0).alpha() == 0
    added, removed = overlay.pixelColor(0, 0), overlay.pixelColor(2, 1)
    assert added.alpha() > 0 and removed.alpha() > 0
    assert added.rgb() != removed.rgb()             # dá para distinguir o pincel da borracha


def test_masks_are_handed_out_as_copies(canvas):
    drag(canvas, (50, 50), (150, 50))

    handed_add, handed_remove = canvas.masks()
    canvas.clear_masks()        # o usuário continua mexendo depois de pedir a restauração

    assert handed_add[100, 200] == 255          # a cópia entregue não muda
    assert handed_add.shape == handed_remove.shape == (200, 400)
    assert not handed_remove.any()


def test_canvas_without_photo_hands_out_no_masks(app):
    assert MaskCanvas().masks() == (None, None)


def test_clearing_the_canvas_removes_the_photo_and_the_strokes(canvas):
    drag(canvas, (50, 50), (150, 50))

    canvas.clear()
    masks_without_photo = canvas.masks()
    canvas.set_photo(np.full((200, 400, 3), BLUE_BGR, np.uint8))        # outra foto, por acaso do mesmo tamanho

    assert masks_without_photo == (None, None)
    assert not canvas.mask_add.any()            # nada da foto anterior


@pytest.fixture
def editor_with_result(app, three_step_result) -> MaskEditor:
    """Editor mostrando um resultado em que a foto original é azul e o mapa de danos é verde."""
    three_step_result.original = solid_photo(BLUE_BGR)
    editor = MaskEditor()
    editor.resize(600, 400)
    editor.set_result(three_step_result)
    return editor


def canvas_color(editor: MaskEditor) -> QColor:
    """A cor no centro da área de pintura."""
    canvas = editor.canvas
    return canvas.grab().toImage().pixelColor(canvas.width() // 2, canvas.height() // 2)


def test_editor_paints_over_the_damage_map_until_the_user_hides_it(editor_with_result):
    with_damage_map = canvas_color(editor_with_result)
    editor_with_result.damage_checkbox.setChecked(False)
    without_damage_map = canvas_color(editor_with_result)
    editor_with_result.damage_checkbox.setChecked(True)

    assert with_damage_map == GREEN
    assert without_damage_map == BLUE
    assert canvas_color(editor_with_result) == GREEN


def test_switching_the_background_keeps_the_strokes(editor_with_result):
    canvas = editor_with_result.canvas
    center = (canvas.width() // 2, canvas.height() // 2)
    drag(canvas, center, center)
    painted = canvas.mask_add.copy()

    editor_with_result.damage_checkbox.setChecked(False)

    assert painted.any() and np.array_equal(canvas.mask_add, painted)


def test_editor_shows_the_original_when_the_result_has_no_damage_map(app, three_step_result):
    three_step_result.original = solid_photo(BLUE_BGR)
    del three_step_result.steps["mapa_de_danos"]
    editor = MaskEditor()
    editor.resize(600, 400)

    editor.set_result(three_step_result)

    assert canvas_color(editor) == BLUE


def test_cleared_editor_stays_empty_when_the_background_is_switched(editor_with_result):
    editor_with_result.clear()

    editor_with_result.damage_checkbox.setChecked(False)

    assert editor_with_result.canvas.masks() == (None, None)
    assert canvas_color(editor_with_result) not in (GREEN, BLUE)


def test_editor_tools_drive_the_canvas(app):
    editor = MaskEditor()
    editor.resize(600, 400)
    editor.canvas.set_photo(np.full((200, 400, 3), RED_BGR, np.uint8))
    center = (editor.canvas.width() // 2, editor.canvas.height() // 2)

    editor.eraser_button.setChecked(True)
    drag(editor.canvas, center, center)
    erased = bool(editor.canvas.mask_remove.any())
    editor.brush_button.setChecked(True)
    editor.size_slider.setValue(40)
    drag(editor.canvas, center, center)
    painted, brush_size = bool(editor.canvas.mask_add.any()), editor.canvas.brush_size
    editor.clear_button.click()

    assert erased and painted
    assert brush_size == 40
    assert not editor.canvas.mask_add.any() and not editor.canvas.mask_remove.any()


# =============================================================== controls.py

def test_untouched_controls_give_the_default_configuration(app):
    assert ControlsPanel().config() == RestorationConfig()


@pytest.mark.parametrize("slider, position, read_field, expected", [
    ("sensitivity", 60, lambda cfg: cfg.damage.sensitivity, 0.6),
    ("denoise_strength", 150, lambda cfg: cfg.preprocess.denoise_strength, 1.5),
    ("clahe_strength", 70, lambda cfg: cfg.preprocess.clahe_strength, 0.7),
    ("sharpen_amount", 90, lambda cfg: cfg.preprocess.sharpen_amount, 0.9),
])
def test_each_slider_sets_its_field_in_hundredths(app, slider, position, read_field, expected):
    panel = ControlsPanel()

    getattr(panel, slider).setValue(position)

    assert read_field(panel.config()) == expected


@pytest.mark.parametrize("checkbox, read_field", [
    ("detect_scratches", lambda cfg: cfg.damage.detect_scratches),
    ("detect_creases", lambda cfg: cfg.damage.detect_creases),
    ("detect_spots", lambda cfg: cfg.damage.detect_spots),
    ("detect_large_losses", lambda cfg: cfg.damage.detect_large_losses),
    ("protect_faces", lambda cfg: cfg.damage.protect_faces),
    ("remove_color_cast", lambda cfg: cfg.preprocess.remove_color_cast),
    ("auto_levels", lambda cfg: cfg.preprocess.auto_levels),
    ("denoise", lambda cfg: cfg.preprocess.denoise),
    ("clahe", lambda cfg: cfg.preprocess.clahe),
    ("sharpen", lambda cfg: cfg.preprocess.sharpen),
])
def test_each_checkbox_switches_its_field(app, checkbox, read_field):
    panel = ControlsPanel()
    default = read_field(RestorationConfig())

    getattr(panel, checkbox).setChecked(not default)

    assert read_field(panel.config()) == (not default)
    assert panel.config() != RestorationConfig()


def test_fill_method_can_be_switched_to_navier_stokes(app):
    panel = ControlsPanel()

    panel.inpaint_method.setCurrentIndex(1)

    assert panel.config().inpaint_method == "ns"


def test_sliders_stop_at_the_limits_the_pipeline_documents(app):
    panel = ControlsPanel()

    for slider in (panel.sensitivity, panel.denoise_strength, panel.clahe_strength, panel.sharpen_amount):
        slider.setValue(100_000)         # o slider para no próprio máximo
    cfg = panel.config()

    assert (cfg.damage.sensitivity, cfg.preprocess.denoise_strength,
            cfg.preprocess.clahe_strength, cfg.preprocess.sharpen_amount) == (1.0, 2.0, 1.0, 1.0)


def paint_it_green(img: np.ndarray, result) -> np.ndarray:
    """Etapa de IA de mentira: fácil de reconhecer no resultado e não depende de modelo baixado."""
    return np.full_like(img, GREEN_BGR)


@pytest.fixture
def colorize_model(monkeypatch):
    """Finge que o modelo de colorização está baixado e troca a rede por ``paint_it_green``.

    Devolve a lista das intensidades pedidas ao criar a etapa.
    """
    requested_strengths = []

    def fake_make_colorize_stage(strength: float):
        requested_strengths.append(strength)
        return paint_it_green

    monkeypatch.setattr(controls_module, "colorize_available", lambda: True)
    monkeypatch.setattr(controls_module, "make_colorize_stage", fake_make_colorize_stage)
    return requested_strengths


def test_colorization_is_off_until_the_user_asks(app, colorize_model):
    assert ControlsPanel().post_stages() == []


def test_checking_colorize_adds_the_stage_with_the_default_strength(app, colorize_model):
    panel = ControlsPanel()

    panel.colorize.setChecked(True)

    assert panel.post_stages() == [("colorizacao", paint_it_green)]
    assert colorize_model == [COLORIZE_DEFAULT_STRENGTH]


def test_colorize_slider_sets_the_strength(app, colorize_model):
    panel = ControlsPanel()
    panel.colorize.setChecked(True)

    panel.colorize_strength.setValue(60)
    panel.post_stages()

    assert colorize_model == [0.6]


def test_panel_announces_every_kind_of_change(app, colorize_model):
    panel = ControlsPanel()
    announcements = []
    panel.changed.connect(lambda: announcements.append("mudou"))

    panel.sensitivity.setValue(70)              # slider
    panel.detect_spots.setChecked(False)        # caixa de marcar
    panel.inpaint_method.setCurrentIndex(1)     # lista
    panel.colorize.setChecked(True)             # etapa de IA
    panel.colorize_strength.setValue(30)

    assert len(announcements) == 5


def test_no_checkbox_text_loses_a_character_to_a_qt_shortcut(app, colorize_model):
    panel = ControlsPanel()

    # No Qt, "&" antes de uma letra vira atalho de teclado e some do texto: "P&B" apareceria "PB".
    shortcuts = {box.text(): QKeySequence.mnemonic(box.text()).toString()
                 for box in panel.findChildren(QCheckBox)}

    assert set(shortcuts.values()) == {""}, shortcuts


def test_colorize_cannot_be_checked_without_the_model(app, monkeypatch):
    monkeypatch.setattr(controls_module, "colorize_available", lambda: False)

    panel = ControlsPanel()

    assert not panel.colorize.isEnabled()
    assert "python -m retrato_vivo modelos" in panel.colorize.toolTip()


# ================================================================= worker.py

def test_worker_passes_the_extra_options_to_restore(app):
    results = []
    worker = RestoreWorker(red_and_blue_photo(), RestorationConfig(), post_stages=[("carimbo", paint_it_green)])
    worker.done.connect(results.append)

    worker.run()

    assert "carimbo" in results[0].steps
    assert np.array_equal(results[0].image, solid_photo(GREEN_BGR))


def test_worker_restores_with_the_given_configuration(app):
    without_sharpening = RestorationConfig()
    without_sharpening.preprocess.sharpen = False
    results = []
    worker = RestoreWorker(red_and_blue_photo(), without_sharpening)
    worker.done.connect(results.append)

    worker.run()

    assert "nitidez" not in results[0].steps
    assert "clahe" in results[0].steps          # as outras etapas continuam


def test_worker_emits_the_restored_photo(app):
    photo = red_and_blue_photo()
    results = []
    worker = RestoreWorker(photo, RestorationConfig())
    worker.done.connect(results.append)

    worker.run()        # chamado direto, sem thread: o sinal chega na mesma hora

    assert len(results) == 1
    assert np.array_equal(results[0].original, photo)
    assert results[0].image.shape == photo.shape


def test_worker_reports_the_error_instead_of_dying(app, broken_restore):
    results, errors = [], []
    worker = RestoreWorker(red_and_blue_photo(), RestorationConfig())
    worker.done.connect(results.append)
    worker.failed.connect(errors.append)

    worker.run()

    assert results == []
    assert len(errors) == 1 and "modelo ausente" in errors[0]


# ============================================================ main_window.py

def test_new_window_says_how_to_start_in_the_middle_of_the_first_tab(window, photo_file):
    window.show()
    QApplication.processEvents()
    first_tab = window.tabs.currentWidget()

    colors_before_any_photo = distinct_colors(first_tab)
    window.open_photo(photo_file)

    assert colors_before_any_photo > 1          # há um texto na área que seria da foto
    shot = first_tab.grab().toImage()
    assert shot.pixelColor(first_tab.width() // 4, first_tab.height() // 2) == RED      # com a foto, o texto some


def test_new_window_shows_a_hint_in_the_status_bar(window):
    assert window.statusBar().currentMessage() != ""


def test_restoration_summary_lists_damage_faces_and_time():
    result = restore(red_and_blue_photo())
    result.damage.stats = {"riscos": 13.0, "vincos": 6.0, "manchas": 77.0,
                           "area_danificada_pct": 2.341, "rostos_protegidos": 2.0}
    result.timings_ms = {"detectar_danos": 400.0, "inpainting": 190.5}

    summary = restoration_summary(result)

    for expected in ("riscos: 13", "vincos: 6", "manchas: 77", "área danificada: 2,34%",
                     "rostos protegidos: 2", "tempo: 0,59 s"):
        assert expected in summary


def test_restoration_summary_without_damage_detection_shows_only_the_time():
    result = restore(red_and_blue_photo())
    result.damage = None
    result.timings_ms = {"abrir": 20.0, "realce": 1230.0}

    assert restoration_summary(result) == "tempo: 1,25 s"


def test_status_bar_shows_the_summary_after_restoring(window, photo_file):
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)

    message = window.statusBar().currentMessage()
    assert "vovó.png" in message
    assert restoration_summary(window.result) in message


def status_bar_notes(window: MainWindow) -> list[str]:
    """Os textos fixos da barra de status (os que não somem quando a mensagem muda)."""
    return [label.text() for label in window.statusBar().findChildren(QLabel)]


def test_changing_a_control_restores_again_by_itself(restored_window):
    window, with_defaults = restored_window, restored_window.result
    window.rerun_timer.setInterval(10)          # nos testes, não precisa esperar o usuário parar de mexer

    window.controls.sharpen.setChecked(False)   # ninguém clica em "Restaurar"
    wait_until(lambda: window.result is not with_defaults)

    assert "nitidez" not in window.result.steps


def test_dragging_a_slider_restores_once_with_the_final_value(restore_calls, restored_window):
    window, with_defaults = restored_window, restored_window.result
    window.rerun_timer.setInterval(50)

    for position in (55, 60, 65, 70, 75):       # o slider passa por vários valores enquanto é arrastado
        window.controls.sensitivity.setValue(position)
    wait_until(lambda: window.result is not with_defaults)
    let_qt_run(0.3)         # tempo para uma restauração a mais aparecer, se existisse

    assert [call["config"].damage.sensitivity for call in restore_calls] == [0.5, 0.75]


def test_change_made_during_a_restoration_is_applied_right_after_it(window, photo_file, monkeypatch):
    real_restore, sharpen_of_each_run, running, most_at_once = worker_module.restore, [], [], []

    def slow_restore(image, config, **options):
        sharpen_of_each_run.append(config.preprocess.sharpen)
        running.append("mais uma")
        most_at_once.append(len(running))
        time.sleep(0.3)         # restauração demorada: dá tempo de o controle mudar com ela em andamento
        result = real_restore(image, config, **options)
        running.pop()
        return result

    monkeypatch.setattr(worker_module, "restore", slow_restore)
    window.rerun_timer.setInterval(10)

    window.open_photo(photo_file)
    window.controls.sharpen.setChecked(False)
    freeze = longest_freeze_until(lambda: len(sharpen_of_each_run) == 2 and window.toolbar.isEnabled())

    assert sharpen_of_each_run == [True, False]
    assert max(most_at_once) == 1               # nunca duas restaurações ao mesmo tempo
    assert freeze < 0.15                        # a janela não ficou parada esperando a primeira terminar
    assert "nitidez" not in window.result.steps


def test_closing_the_window_cancels_the_pending_restoration(restore_calls, restored_window):
    restored_window.rerun_timer.setInterval(50)
    restored_window.controls.sharpen.setChecked(False)

    restored_window.close()
    let_qt_run(0.25)

    assert len(restore_calls) == 1              # só a restauração de quando a foto foi aberta


def test_changing_controls_without_a_photo_does_nothing(window):
    window.rerun_timer.setInterval(10)

    window.controls.sharpen.setChecked(False)
    let_qt_run(0.2)

    assert window.worker is None
    assert window.result is None


def test_restore_button_applies_the_current_controls(window, photo_file):
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)
    with_defaults = window.result

    window.controls.sharpen.setChecked(False)
    window.restore_action.trigger()
    wait_until(lambda: window.result is not with_defaults)

    assert "nitidez" in with_defaults.steps
    assert "nitidez" not in window.result.steps


def test_restore_button_runs_the_ai_stages_chosen_in_the_controls(colorize_model, window, photo_file):
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)
    without_colorization = window.result

    window.controls.colorize.setChecked(True)
    window.restore_action.trigger()
    wait_until(lambda: window.result is not without_colorization)

    assert "colorizacao" not in without_colorization.steps
    assert "colorizacao" in window.result.steps
    assert np.array_equal(window.result.image, solid_photo(GREEN_BGR))


@pytest.mark.skipif(not colorize_model_is_downloaded(),
                    reason="modelo de colorização não baixado (python -m retrato_vivo modelos)")
def test_colorize_control_really_colors_a_gray_photo(window, tmp_path):
    gray_ramp = np.tile(np.linspace(40, 220, 200).astype(np.uint8)[None, :, None], (100, 1, 3))
    gray_file = save_image(tmp_path / "cinza.png", gray_ramp)
    window.controls.colorize.setChecked(True)

    window.open_photo(gray_file)
    wait_until(lambda: window.result is not None, timeout_s=60)

    blue, green, red = (window.result.image[..., channel].astype(int) for channel in range(3))
    assert "colorida pela rede" in window.result.notes["colorizacao"]
    assert np.abs(red - blue).max() > 0        # numa foto cinza os três canais seriam iguais


@pytest.fixture
def restore_calls(monkeypatch):
    """Anota a configuração e os argumentos nomeados de cada chamada a ``restore()``, que acontece de verdade."""
    calls = []
    real_restore = worker_module.restore

    def recording_restore(image, config, **options):
        calls.append({"config": config, **options})
        return real_restore(image, config, **options)

    monkeypatch.setattr(worker_module, "restore", recording_restore)
    return calls


def paint_on_the_mask_tab(window: MainWindow, x_fraction: float) -> None:
    """Um clique na área de pintura, na altura do meio e a ``x_fraction`` da largura."""
    canvas = window.mask_editor.canvas
    position = (round(canvas.width() * x_fraction), canvas.height() // 2)
    drag(canvas, position, position)


def test_apply_button_restores_again_with_the_painted_masks(restore_calls, window, photo_file):
    window.resize(1000, 700)
    window.show()
    window.tabs.setCurrentWidget(window.mask_editor)
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)
    first = window.result
    canvas = window.mask_editor.canvas

    paint_on_the_mask_tab(window, 0.25)                 # pincel na metade vermelha: (50, 50) na foto
    window.mask_editor.eraser_button.setChecked(True)
    paint_on_the_mask_tab(window, 0.75)                 # borracha na metade azul: (150, 50) na foto
    window.mask_editor.apply_button.click()
    wait_until(lambda: window.result is not first)

    on_open, on_apply = restore_calls
    assert on_open["mask_add"] is None and on_open["mask_remove"] is None
    assert on_apply["mask_add"][50, 50] == 255 and np.array_equal(on_apply["mask_add"], canvas.mask_add)
    assert on_apply["mask_remove"][50, 150] == 255 and np.array_equal(on_apply["mask_remove"], canvas.mask_remove)
    assert first.damage.mask[50, 50] == 0               # a detecção automática não tinha marcado ali
    assert window.result.damage.mask[50, 50] == 255     # agora o reparo passa por onde o usuário pintou


def test_apply_button_needs_an_open_photo_and_an_idle_window(window, photo_file):
    before_any_photo = window.mask_editor.apply_button.isEnabled()
    window.open_photo(photo_file)
    while_restoring = window.mask_editor.apply_button.isEnabled()
    wait_until(lambda: window.result is not None)

    assert not before_any_photo
    assert not while_restoring
    assert window.mask_editor.apply_button.isEnabled()


def test_opening_another_photo_discards_the_painted_masks(restore_calls, window, photo_file, tmp_path):
    green_file = save_image(tmp_path / "verde.png", solid_photo(GREEN_BGR))     # do mesmo tamanho da primeira
    window.resize(1000, 700)
    window.show()
    window.tabs.setCurrentWidget(window.mask_editor)
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)
    paint_on_the_mask_tab(window, 0.25)

    window.open_photo(green_file)
    wait_until(lambda: window.result is not None)

    assert restore_calls[-1]["mask_add"] is None        # a foto nova foi restaurada sem a pintura da anterior
    assert not window.mask_editor.canvas.mask_add.any()


def test_smallest_window_still_fits_the_controls_and_the_mask_tools(window):
    window.show()
    window.tabs.setCurrentWidget(window.mask_editor)
    window.resize(1, 1)                 # o Qt não deixa a janela ficar menor do que o layout precisa
    QApplication.processEvents()        # deixa o Qt recalcular o layout
    editor = window.mask_editor

    visible_width = window.controls.visibleRegion().boundingRect().width()
    assert visible_width == window.controls.width()                         # painel inteiro à vista, sem rolar para o lado
    assert editor.size_slider.width() >= 80                                 # dá para arrastar
    assert editor.apply_button.geometry().right() <= editor.width()         # nada fora da aba
    assert window.width() <= 1024                                           # ainda cabe num projetor antigo


def test_mask_tab_shows_what_was_detected_in_the_restored_photo(window, photo_file):
    window.resize(1000, 700)
    window.show()
    window.tabs.setCurrentWidget(window.mask_editor)
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)
    canvas = window.mask_editor.canvas
    left_quarter = (canvas.width() // 4, canvas.height() // 2)      # (50, 50) na foto de 200x100

    with_damage_map = canvas.grab().toImage().pixelColor(*left_quarter)
    window.mask_editor.damage_checkbox.setChecked(False)
    without_damage_map = canvas.grab().toImage().pixelColor(*left_quarter)

    blue, green, red = (int(channel) for channel in window.result.steps["mapa_de_danos"][50, 50])
    assert with_damage_map == QColor(red, green, blue)
    assert without_damage_map == RED            # a foto original
    assert with_damage_map != without_damage_map


def test_mask_tab_shows_the_restored_photo_ready_to_paint(window, photo_file):
    tab_titles = [window.tabs.tabText(index) for index in range(window.tabs.count())]
    window.resize(1000, 700)
    window.show()
    window.tabs.setCurrentWidget(window.mask_editor)

    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)
    canvas = window.mask_editor.canvas
    center = (canvas.width() // 2, canvas.height() // 2)
    drag(canvas, center, center)

    assert "Máscara" in tab_titles
    assert canvas.mask_add.shape == window.result.original.shape[:2]
    assert canvas.mask_add[50, 100] == 255          # o centro da foto de 200x100


@pytest.fixture
def save_dialog(monkeypatch):
    """Troca o diálogo de salvar por uma resposta pronta. Anota o nome que a janela sugeriu a cada chamada."""
    dialog = types.SimpleNamespace(answer="", suggestions=[])

    def fake_dialog(parent, title, suggestion, filters):
        dialog.suggestions.append(suggestion)
        return str(dialog.answer), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_dialog)
    return dialog


@pytest.fixture
def restored_window(window, photo_file) -> MainWindow:
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)
    return window


def test_save_actions_need_a_restored_photo(window, photo_file, tmp_path):
    green_file = save_image(tmp_path / "verde.png", solid_photo(GREEN_BGR))
    before_any_photo = (window.save_action.isEnabled(), window.export_action.isEnabled())
    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)
    after_restoring = (window.save_action.isEnabled(), window.export_action.isEnabled())
    window.open_photo(green_file)
    with_a_new_photo_not_restored_yet = (window.save_action.isEnabled(), window.export_action.isEnabled())

    assert before_any_photo == (False, False)
    assert after_restoring == (True, True)
    assert with_a_new_photo_not_restored_yet == (False, False)


def test_saving_the_result_writes_the_restored_photo(restored_window, save_dialog, tmp_path):
    save_dialog.answer = tmp_path / "saída" / "restaurada.png"

    restored_window.save_action.trigger()

    assert np.array_equal(load_image(save_dialog.answer), restored_window.result.image)
    assert save_dialog.suggestions[0].endswith("vovó_restaurada.png")
    assert "restaurada.png" in restored_window.statusBar().currentMessage()


def test_exporting_the_comparison_writes_before_and_after_side_by_side(restored_window, save_dialog, tmp_path):
    save_dialog.answer = tmp_path / "comparativo.png"
    result = restored_window.result

    restored_window.export_action.trigger()

    assert np.array_equal(load_image(save_dialog.answer), side_by_side(result.original, result.image))
    assert save_dialog.suggestions[0].endswith("vovó_antes_depois.jpg")


def test_cancelling_the_save_dialog_writes_nothing(restored_window, save_dialog, tmp_path, warnings):
    files_before = sorted(tmp_path.iterdir())
    message_before = restored_window.statusBar().currentMessage()

    restored_window.save_action.trigger()       # o diálogo de mentira devolve "", como no botão Cancelar

    assert sorted(tmp_path.iterdir()) == files_before
    assert restored_window.statusBar().currentMessage() == message_before
    assert warnings == []                       # cancelar não é erro


@pytest.mark.parametrize("bad_name", [
    "foto.formato_que_nao_existe",      # o OpenCV não sabe gravar esta extensão
    "vovó.png/dentro_de_um_arquivo.png",    # a "pasta" é a própria foto de teste
])
def test_failed_save_warns_and_does_not_claim_success(restored_window, save_dialog, tmp_path, warnings, bad_name):
    save_dialog.answer = tmp_path / bad_name

    restored_window.save_action.trigger()

    assert len(warnings) == 1 and bad_name.split("/")[-1] in warnings[0]
    assert "Salvo" not in restored_window.statusBar().currentMessage()


def test_steps_tab_follows_the_open_photo(window, photo_file, tmp_path):
    green_file = save_image(tmp_path / "verde.png", solid_photo(GREEN_BGR))
    tab_titles = [window.tabs.tabText(index) for index in range(window.tabs.count())]

    window.open_photo(photo_file)
    wait_until(lambda: window.result is not None)
    restored_steps = len(window.result.steps)
    listed_after_restoring = window.steps_view.step_list.count()
    window.open_photo(green_file)
    listed_right_after_opening_another = window.steps_view.step_list.count()

    assert "Etapas" in tab_titles
    assert listed_after_restoring == restored_steps > 0
    assert listed_right_after_opening_another == 0      # as etapas antigas eram de outra foto


def test_window_tells_how_to_download_a_missing_face_model(app, monkeypatch):
    monkeypatch.setattr(models, "is_available", lambda name: name != "yunet")
    without_model = MainWindow()
    monkeypatch.setattr(models, "is_available", lambda name: True)
    with_model = MainWindow()

    assert any("python -m retrato_vivo modelos" in note for note in status_bar_notes(without_model))
    assert status_bar_notes(with_model) == []


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
