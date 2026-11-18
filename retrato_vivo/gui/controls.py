"""Painel de controles: os ajustes que o usuário pode fazer na restauração.

Cada controle tem o nome do campo de ``RestorationConfig`` que ele altera. A lista
vem da seção 5 do ``docs/COMO_FUNCIONA.md``. As etapas de IA (``retrato_vivo/ai``)
não são campos da configuração: entram por ``post_stages()``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QSlider, QVBoxLayout, QWidget

from ..ai.colorize import DEFAULT_STRENGTH, make_colorize_stage
from ..ai.colorize import is_available as colorize_available
from ..pipeline import PostStage, RestorationConfig

SLIDER_STEPS = 100   # o QSlider só conhece inteiros: cada passo vale 0,01


def make_slider(maximum: float, value: float) -> QSlider:
    """Slider de 0 a ``maximum``, começando em ``value``."""
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, round(maximum * SLIDER_STEPS))
    slider.setValue(round(value * SLIDER_STEPS))
    return slider


def slider_value(slider: QSlider) -> float:
    return slider.value() / SLIDER_STEPS


def make_checkbox(text: str, checked: bool) -> QCheckBox:
    checkbox = QCheckBox(text)
    checkbox.setChecked(checked)
    return checkbox


class ControlsPanel(QWidget):
    """Os controles começam nos valores padrão do pipeline; ``config()`` lê o estado atual."""

    changed = Signal()     # o usuário mexeu em algum controle

    def __init__(self) -> None:
        super().__init__()
        defaults = RestorationConfig()
        damage, preprocess = defaults.damage, defaults.preprocess

        self.sensitivity = make_slider(1.0, damage.sensitivity)
        self.detect_scratches = make_checkbox("Riscos", damage.detect_scratches)
        self.detect_creases = make_checkbox("Vincos", damage.detect_creases)
        self.detect_spots = make_checkbox("Manchas", damage.detect_spots)
        self.detect_large_losses = make_checkbox("Rasgos e áreas brancas", damage.detect_large_losses)
        self.protect_faces = make_checkbox("Proteger rostos", damage.protect_faces)
        detection = QGroupBox("Detecção de danos")
        form = QFormLayout(detection)
        form.addRow("Sensibilidade", self.sensitivity)
        for checkbox in (self.detect_scratches, self.detect_creases, self.detect_spots,
                         self.detect_large_losses, self.protect_faces):
            form.addRow(checkbox)

        self.inpaint_method = QComboBox()
        self.inpaint_method.addItem("Telea (rápido)", "telea")
        self.inpaint_method.addItem("Navier-Stokes (áreas maiores)", "ns")
        self.inpaint_method.setCurrentIndex(self.inpaint_method.findData(defaults.inpaint_method))
        fill = QGroupBox("Preenchimento dos danos")
        QFormLayout(fill).addRow("Método", self.inpaint_method)

        self.remove_color_cast = make_checkbox("Tirar amarelado ou sépia", preprocess.remove_color_cast)
        self.auto_levels = make_checkbox("Recuperar contraste", preprocess.auto_levels)
        self.denoise = make_checkbox("Reduzir ruído", preprocess.denoise)
        self.denoise_strength = make_slider(2.0, preprocess.denoise_strength)
        self.clahe = make_checkbox("Contraste local", preprocess.clahe)
        self.clahe_strength = make_slider(1.0, preprocess.clahe_strength)
        self.sharpen = make_checkbox("Nitidez", preprocess.sharpen)
        self.sharpen_amount = make_slider(1.0, preprocess.sharpen_amount)
        corrections = QGroupBox("Correções")
        form = QFormLayout(corrections)
        form.addRow(self.remove_color_cast)
        form.addRow(self.auto_levels)
        form.addRow(self.denoise, self.denoise_strength)     # a caixa liga a correção, o slider dá a intensidade
        form.addRow(self.clahe, self.clahe_strength)
        form.addRow(self.sharpen, self.sharpen_amount)

        # "&&" é como o Qt escreve um "&" de verdade; um só viraria atalho de teclado e sumiria do texto.
        self.colorize = make_checkbox("Colorir foto P&&B", False)
        self.colorize_strength = make_slider(1.0, DEFAULT_STRENGTH)
        if not colorize_available():
            self.colorize.setEnabled(False)
            self.colorize.setText("Colorir (modelo não baixado)")
            self.colorize.setToolTip("Rode: python -m retrato_vivo modelos")
        deep_learning = QGroupBox("Deep Learning")
        QFormLayout(deep_learning).addRow(self.colorize, self.colorize_strength)

        layout = QVBoxLayout(self)
        for group in (detection, fill, corrections, deep_learning):
            layout.addWidget(group)
        layout.addStretch()

        # Qualquer mexida avisa a janela, que refaz a restauração sozinha. Procurar os controles
        # pelo tipo faz um controle novo já entrar nesta lista.
        for slider in self.findChildren(QSlider):
            slider.valueChanged.connect(self._announce_change)
        for checkbox in self.findChildren(QCheckBox):
            checkbox.toggled.connect(self._announce_change)
        self.inpaint_method.currentIndexChanged.connect(self._announce_change)

    def _announce_change(self) -> None:
        self.changed.emit()

    def post_stages(self) -> list[tuple[str, PostStage]]:
        """As etapas de IA ligadas, no formato de ``restore(post_stages=...)``."""
        stages = []
        if self.colorize.isChecked():
            stages.append(("colorizacao", make_colorize_stage(slider_value(self.colorize_strength))))
        return stages

    def config(self) -> RestorationConfig:
        """A configuração que os controles pedem agora. O que não tem controle fica no padrão."""
        cfg = RestorationConfig(inpaint_method=self.inpaint_method.currentData())

        cfg.damage.sensitivity = slider_value(self.sensitivity)
        cfg.damage.detect_scratches = self.detect_scratches.isChecked()
        cfg.damage.detect_creases = self.detect_creases.isChecked()
        cfg.damage.detect_spots = self.detect_spots.isChecked()
        cfg.damage.detect_large_losses = self.detect_large_losses.isChecked()
        cfg.damage.protect_faces = self.protect_faces.isChecked()

        cfg.preprocess.remove_color_cast = self.remove_color_cast.isChecked()
        cfg.preprocess.auto_levels = self.auto_levels.isChecked()
        cfg.preprocess.denoise = self.denoise.isChecked()
        cfg.preprocess.denoise_strength = slider_value(self.denoise_strength)
        cfg.preprocess.clahe = self.clahe.isChecked()
        cfg.preprocess.clahe_strength = slider_value(self.clahe_strength)
        cfg.preprocess.sharpen = self.sharpen.isChecked()
        cfg.preprocess.sharpen_amount = slider_value(self.sharpen_amount)
        return cfg
