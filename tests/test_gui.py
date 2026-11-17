"""Testes da interface gráfica. Rodar com ``python -m pytest``.

Rodam sem tela: a plataforma "offscreen" do Qt desenha as janelas na memória.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from retrato_vivo.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    """Qt exige uma única QApplication viva antes de criar qualquer widget."""
    return QApplication.instance() or QApplication([])


# ============================================================ main_window.py

def test_new_window_shows_a_hint_in_the_status_bar(app):
    window = MainWindow()

    assert window.statusBar().currentMessage() != ""
