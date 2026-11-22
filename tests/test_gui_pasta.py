"""Testes da navegação pelas fotos da pasta (``gui/folder_nav.py``)."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from retrato_vivo.gui import worker as worker_module  # noqa: E402
from retrato_vivo.gui.folder_nav import PhotoFolder, first_photo  # noqa: E402
from retrato_vivo.gui.main_window import MainWindow  # noqa: E402
from retrato_vivo.io_utils import save_image  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def folder(tmp_path):
    """Pasta com três fotos pequenas e um arquivo que não é foto."""
    for name, value in (("a.png", 50), ("b.jpg", 120), ("c.png", 200)):
        save_image(tmp_path / name, np.full((40, 60, 3), value, np.uint8))
    (tmp_path / "notas.txt").write_text("não é foto")
    return tmp_path


@pytest.fixture
def window(app, monkeypatch):
    # Restauração instantânea: estes testes são sobre navegação, não sobre o pipeline.
    monkeypatch.setattr(worker_module.RestoreWorker, "start", lambda self: self.run())
    w = MainWindow()
    yield w
    w.close()


def wait_until(condition, timeout_s=3.0):
    end = time.time() + timeout_s
    while time.time() < end:
        QApplication.processEvents()
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("condição não foi atingida a tempo")


def test_first_photo(folder, tmp_path_factory):
    assert first_photo(folder).name == "a.png"
    assert first_photo(tmp_path_factory.mktemp("vazia")) is None
    assert first_photo(folder / "nao_existe") is None


def test_neighbor_wraps_around(app, folder):
    pf = PhotoFolder()
    pf.follow(folder / "a.png")
    assert [p.name for p in pf.photos] == ["a.png", "b.jpg", "c.png"]       # o .txt fica de fora
    assert pf.neighbor(folder / "a.png", +1).name == "b.jpg"
    assert pf.neighbor(folder / "c.png", +1).name == "a.png"                # dá a volta
    assert pf.neighbor(folder / "a.png", -1).name == "c.png"
    assert pf.neighbor(folder / "sumiu.png", +1).name == "a.png"            # foto removida: recomeça


def test_window_walks_through_folder(window, folder):
    window.open_photo(folder / "a.png")
    wait_until(lambda: window.result is not None)
    assert window.photo_picker.count() == 3 and window.photo_picker.currentText() == "a.png"
    window.next_action.trigger()
    wait_until(lambda: window.photo_path.name == "b.jpg" and window.result is not None)
    assert window.photo_picker.currentText() == "b.jpg"
    window.previous_action.trigger()
    window.previous_action.trigger()
    wait_until(lambda: window.photo_path.name == "c.png")


def test_picker_opens_chosen_photo(window, folder):
    window.open_photo(folder / "a.png")
    window.photo_picker.activated.emit(2)
    wait_until(lambda: window.photo_path.name == "c.png")


def test_list_updates_when_photo_is_added(window, folder):
    window.open_photo(folder / "a.png")
    save_image(folder / "d.png", np.zeros((40, 60, 3), np.uint8))
    wait_until(lambda: window.photo_picker.count() == 4)
    assert window.photo_picker.itemText(3) == "d.png"


def test_navigation_disabled_without_photo(window):
    assert window.photo_picker.count() == 0
    assert not window.next_action.isEnabled() and not window.previous_action.isEnabled()
