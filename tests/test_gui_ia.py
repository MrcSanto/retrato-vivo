"""Testes da interface para a super-resolução (opção "Ampliar") e a aba "Bordas".

Rodam sem tela (plataforma "offscreen" do Qt) e sem modelos baixados: as redes são
trocadas por funções de mentira.
"""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from retrato_vivo.gui import controls as controls_module
from retrato_vivo.gui import edges_view as edges_view_module
from retrato_vivo.gui.controls import ControlsPanel
from retrato_vivo.gui.edges_view import EdgesView
from retrato_vivo.gui.main_window import MainWindow
from retrato_vivo.pipeline import RestorationResult


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(condition, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "a condição não se cumpriu a tempo"
        QApplication.processEvents()
        time.sleep(0.005)


def let_qt_run(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)


def fake_result(value: int, shape=(60, 80)) -> RestorationResult:
    original = np.full((*shape, 3), value, np.uint8)
    return RestorationResult(image=original.copy(), original=original, scale=1.0, diagnosis=None)


# ================================================================ "Ampliar"

@pytest.fixture
def upscale_model(monkeypatch):
    """Finge que os modelos de super-resolução estão baixados; devolve os modelos pedidos."""
    requested = []

    def fake_make_upscale_stage(model="espcn", target=2048):
        requested.append(model)
        return f"etapa_{model}"

    monkeypatch.setattr(controls_module.superres, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(controls_module.superres, "make_upscale_stage", fake_make_upscale_stage)
    return requested


def test_upscale_is_off_until_the_user_asks(app, upscale_model):
    assert ControlsPanel().post_stages() == []


def test_upscale_uses_espcn_by_default_and_edsr_when_chosen(app, upscale_model):
    panel = ControlsPanel()
    panel.upscale.setChecked(True)
    assert panel.post_stages() == [("super_resolucao", "etapa_espcn")]

    panel.upscale_model.setCurrentIndex(panel.upscale_model.findData("edsr"))
    assert panel.post_stages() == [("super_resolucao", "etapa_edsr")]
    assert upscale_model == ["espcn", "edsr"]


def test_upscale_comes_after_colorization(app, upscale_model, monkeypatch):
    monkeypatch.setattr(controls_module, "colorize_available", lambda: True)
    monkeypatch.setattr(controls_module, "make_colorize_stage", lambda strength: "etapa_cor")
    monkeypatch.setattr(controls_module.ddcolor, "is_available", lambda: False)
    panel = ControlsPanel()
    panel.colorize.setChecked(True)
    panel.upscale.setChecked(True)
    assert [name for name, _ in panel.post_stages()] == ["colorizacao", "super_resolucao"]


def test_changing_the_upscale_model_redoes_the_restoration(app, upscale_model):
    panel = ControlsPanel()
    announcements = []
    panel.changed.connect(lambda: announcements.append("mudou"))
    panel.upscale.setChecked(True)
    panel.upscale_model.setCurrentIndex(1)
    assert len(announcements) == 2


def test_upscale_cannot_be_checked_without_the_model(app, monkeypatch):
    monkeypatch.setattr(controls_module.superres, "is_available", lambda *a, **k: False)
    panel = ControlsPanel()
    assert not panel.upscale.isEnabled()
    assert "sem modelo" in panel.upscale.text()


# ================================================================== "Bordas"

@pytest.fixture
def fake_panel(monkeypatch):
    """Troca o painel de bordas por um que devolve o "antes" e anota as chamadas."""
    calls = []

    def painel_bordas(before, after):
        calls.append(int(before[0, 0, 0]))
        time.sleep(0.05)                # dá tempo de chegar outra restauração no meio
        return before.copy()

    monkeypatch.setattr(edges_view_module.edges, "painel_bordas", painel_bordas)
    monkeypatch.setattr(edges_view_module.edges, "is_available", lambda: True)
    return calls


def test_edges_tab_exists_in_the_window(app):
    window = MainWindow()
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert "Bordas" in titles
    assert not window.edges_view.save_button.isEnabled()
    window.close()


def test_edges_are_only_computed_when_the_tab_is_visible(app, fake_panel):
    view = EdgesView()
    view.set_result(fake_result(10))
    let_qt_run(0.1)
    assert fake_panel == []                     # escondida: não gasta o HED

    view.show()
    wait_until(lambda: view.panel is not None)
    assert fake_panel == [10]
    assert view.save_button.isEnabled()
    view.shutdown()
    view.close()


def test_a_new_restoration_while_computing_shows_the_newest_panel(app, fake_panel):
    view = EdgesView()
    view.show()
    view.set_result(fake_result(10))
    view.set_result(fake_result(20))            # chega antes de o primeiro painel ficar pronto
    wait_until(lambda: view.panel is not None and view.panel[0, 0, 0] == 20)
    assert fake_panel[-1] == 20
    view.shutdown()
    view.close()


def test_opening_another_photo_clears_the_edges(app, fake_panel):
    view = EdgesView()
    view.show()
    view.set_result(fake_result(10))
    wait_until(lambda: view.panel is not None)
    view.set_result(None)
    assert view.panel is None and not view.save_button.isEnabled()
    view.shutdown()
    view.close()


def test_edges_run_off_the_interface_thread(app, fake_panel, monkeypatch):
    threads = []
    original = edges_view_module.edges.painel_bordas

    def spy(before, after):
        threads.append(threading.current_thread() is threading.main_thread())
        return original(before, after)

    monkeypatch.setattr(edges_view_module.edges, "painel_bordas", spy)
    view = EdgesView()
    view.show()
    view.set_result(fake_result(30))
    wait_until(lambda: view.panel is not None)
    assert threads == [False]
    view.shutdown()
    view.close()
