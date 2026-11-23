"""Testes das abas do diferencial na interface: "Retrato Vivo" e "Máquina do tempo".

Rodam sem tela (plataforma "offscreen" do Qt) e sem modelos baixados: a criação do
retrato e a restauração são trocadas por trabalhos de mentira, que só terminam quando
o teste manda. Assim dá para mexer na interface no meio do trabalho, que é quando os
defeitos destas abas apareciam.
"""

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from retrato_vivo import living_portrait as lp
from retrato_vivo.gui import extras as extras_module
from retrato_vivo.gui.extras import LivingPortraitView, TimeMachineView
from retrato_vivo.gui.main_window import MainWindow
from retrato_vivo.pipeline import RestorationResult

from test_gui import let_qt_run, wait_until

WORK_TIMEOUT_S = 2.0     # o trabalho de mentira desiste de esperar o teste depois disto
RESTORED_VALUE = 77      # a restauração de mentira devolve uma foto lisa com este valor


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def photo(value: int, shape=(60, 80)) -> np.ndarray:
    return np.full((*shape, 3), value, np.uint8)


class BackgroundWork:
    """Trabalho demorado de mentira: só termina quando o teste chama ``finish()``."""

    def __init__(self) -> None:
        self._may_finish = threading.Event()
        self.ran_on_main_thread: list[bool] = []

    def wait_for_the_test(self) -> None:
        self.ran_on_main_thread.append(threading.current_thread() is threading.main_thread())
        self._may_finish.wait(WORK_TIMEOUT_S)

    def finish(self) -> None:
        self._may_finish.set()


def finish_and_deliver(work: BackgroundWork, view) -> None:
    """Deixa o trabalho de mentira terminar e o Qt entregar o resultado à interface."""
    work.finish()
    view.worker.wait()
    let_qt_run(0.05)


# ================================================================ Retrato Vivo

@pytest.fixture
def portrait_work(monkeypatch):
    """Troca a criação do Retrato Vivo por uma de mentira, que espera o teste mandar terminar."""
    work = BackgroundWork()

    def make_living_portrait(source: np.ndarray) -> lp.LivingPortrait:
        work.wait_for_the_test()
        flat = np.zeros(source.shape[:2], np.float32)
        return lp.LivingPortrait(frames=[source.copy()], depth=flat, foreground=flat,
                                 background=source.copy(), fps=25, timings_ms={"total": 1.0}, device="CPU")

    monkeypatch.setattr(extras_module.lp, "is_available", lambda: True)
    monkeypatch.setattr(extras_module.lp, "make_living_portrait", make_living_portrait)
    yield work
    work.finish()


@pytest.fixture
def living_view(app, portrait_work):
    view = LivingPortraitView()
    yield view
    portrait_work.finish()
    view.shutdown()
    view.close()


def test_portrait_of_the_open_photo_is_shown_when_ready(living_view, portrait_work):
    living_view.set_source(photo(10))
    living_view.create_button.click()

    finish_and_deliver(portrait_work, living_view)

    assert living_view.portrait is not None and living_view.portrait.frames[0][0, 0, 0] == 10
    assert living_view.save_video_button.isEnabled()


def test_portrait_of_a_previous_photo_is_discarded_after_opening_another(living_view, portrait_work):
    living_view.set_source(photo(10))
    status_before_creating = living_view.status.text()
    living_view.create_button.click()
    living_view.set_source(photo(200))           # outra foto chega no meio da criação

    finish_and_deliver(portrait_work, living_view)

    assert living_view.portrait is None          # o retrato que ficou pronto era da foto anterior
    assert not living_view.save_video_button.isEnabled()
    assert living_view.status.text() == status_before_creating     # nada de "criando…" para sempre


def test_create_is_locked_until_the_running_creation_ends(living_view, portrait_work):
    living_view.set_source(photo(10))
    living_view.create_button.click()
    living_view.set_source(photo(200))

    # Um segundo clique agora esperaria a thread anterior terminar, com a janela travada.
    assert not living_view.create_button.isEnabled()

    finish_and_deliver(portrait_work, living_view)
    assert living_view.create_button.isEnabled()


# ============================================================ Máquina do tempo

@pytest.fixture
def restore_work(monkeypatch):
    """Troca a restauração por uma de mentira, que espera o teste mandar terminar."""
    work = BackgroundWork()

    def restore(aged: np.ndarray) -> RestorationResult:
        work.wait_for_the_test()
        return RestorationResult(image=np.full_like(aged, RESTORED_VALUE), original=aged, scale=1.0,
                                 diagnosis=None)

    monkeypatch.setattr(extras_module, "restore", restore)
    yield work
    work.finish()


@pytest.fixture
def time_machine(app, restore_work):
    """A aba com uma foto aberta e já envelhecida para 1920."""
    view = TimeMachineView()
    view.set_source(photo(120))
    view.era_buttons["1920"].click()
    yield view
    restore_work.finish()
    view.shutdown()
    view.close()


def test_restore_back_runs_off_the_interface_thread(time_machine, restore_work):
    restore_work.finish()                        # sem espera: aqui só importa em que thread rodou
    aged = time_machine.aged

    time_machine.restore_back.click()
    wait_until(lambda: time_machine.shown is not aged)

    assert restore_work.ran_on_main_thread == [False]


def test_restore_back_shows_the_aged_photo_beside_the_restored_one(time_machine, restore_work):
    restore_work.finish()
    aged = time_machine.aged
    width = aged.shape[1]

    time_machine.restore_back.click()
    wait_until(lambda: time_machine.shown is not aged)

    shown = time_machine.shown
    assert shown.shape[1] > 2 * width                           # as duas fotos e o espaço entre elas
    assert np.array_equal(shown[:, :width], aged)               # esquerda: a envelhecida
    assert (shown[:, -width:] == RESTORED_VALUE).all()          # direita: a restaurada


def test_restore_back_is_locked_while_restoring(time_machine, restore_work):
    time_machine.restore_back.click()

    # Um segundo clique agora começaria outra restauração por cima da primeira.
    assert not time_machine.restore_back.isEnabled()

    finish_and_deliver(restore_work, time_machine)
    assert time_machine.restore_back.isEnabled()


def test_restored_photo_is_discarded_after_opening_another_photo(time_machine, restore_work):
    time_machine.restore_back.click()
    time_machine.set_source(photo(30))           # outra foto é aberta no meio da restauração

    finish_and_deliver(restore_work, time_machine)

    assert time_machine.shown is None            # a foto nova ainda não viajou no tempo
    assert not time_machine.save_button.isEnabled()


def test_choosing_another_era_while_restoring_keeps_the_new_era_on_screen(time_machine, restore_work):
    time_machine.restore_back.click()
    time_machine.era_buttons["1950"].click()     # o usuário muda de ideia no meio da restauração
    newest = time_machine.aged

    finish_and_deliver(restore_work, time_machine)

    assert time_machine.shown is newest
    assert time_machine.restore_back.isEnabled()


def test_a_failed_restore_back_is_reported_and_can_be_tried_again(time_machine, monkeypatch):
    def broken_restore(aged: np.ndarray) -> RestorationResult:
        raise RuntimeError("modelo corrompido")

    monkeypatch.setattr(extras_module, "restore", broken_restore)
    aged = time_machine.aged

    time_machine.restore_back.click()
    wait_until(lambda: "modelo corrompido" in time_machine.status.text(), timeout_s=2.0)

    assert time_machine.shown is aged            # a foto envelhecida continua na tela
    assert time_machine.restore_back.isEnabled()


def test_closing_the_window_waits_for_restore_back(app, restore_work):
    window = MainWindow()
    window.time_machine.set_source(photo(120))
    window.time_machine.era_buttons["1920"].click()
    window.time_machine.restore_back.click()
    threading.Timer(0.1, restore_work.finish).start()

    try:
        window.close()
        # O Qt aborta o programa se um QThread for destruído enquanto ainda roda.
        assert window.time_machine.worker.isFinished()
    finally:
        restore_work.finish()
        window.time_machine.worker.wait()


def test_restore_back_requested_twice_restores_only_once(time_machine, restore_work):
    time_machine.round_trip()
    time_machine.round_trip()                    # pedido por código, sem passar pelo botão travado

    finish_and_deliver(restore_work, time_machine)

    assert restore_work.ran_on_main_thread == [False]       # uma restauração só, e fora da interface
