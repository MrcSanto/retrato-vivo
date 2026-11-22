"""Painel de controles: os ajustes que o usuário pode fazer na restauração.

Cada controle tem o nome do campo de ``RestorationConfig`` que ele altera. A lista
vem da seção 5 do ``docs/COMO_FUNCIONA.md``. As etapas de IA (``retrato_vivo/ai``)
não são campos da configuração: entram por ``post_stages()``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QLabel, QSlider, QToolButton,
                               QVBoxLayout, QWidget)

from .. import damage_ai, inpaint
from ..ai import ddcolor, face_restore, superres
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
        self.use_ai = make_checkbox("Riscos com IA", damage.use_ai)
        if not damage_ai.is_available():
            self.use_ai.setEnabled(False)
            self.use_ai.setText("Riscos com IA (sem modelo)")
        self.use_ai.setToolTip("Rede treinada em fotos antigas reais (Microsoft). Pega riscos que as regras perdem.")
        # Escondidos (diagnóstico de 20/11): "Rasgos" quase nunca age e confunde fundos claros com
        # rasgos; rasgo se marca com o pincel da aba Máscara. "Proteger rostos" fica sempre ligado.
        # Continuam existindo para quem usa o painel por código.
        self.detect_large_losses.hide()
        self.protect_faces.hide()

        self.inpaint_method = QComboBox()
        self.inpaint_method.addItem("Telea (rápido)", "telea")
        self.inpaint_method.addItem("Navier-Stokes (áreas maiores)", "ns")
        self.inpaint_method.addItem("IA (LaMa)" if inpaint.lama_available() else "IA (sem modelo)", "ia")
        self.inpaint_method.setToolTip("Com IA, riscos finos usam o Telea e áreas grandes (pintadas com o "
                                       "pincel) são reconstruídas pelo LaMa.")
        self.inpaint_method.setCurrentIndex(self.inpaint_method.findData(defaults.inpaint_method))

        self.remove_color_cast = make_checkbox("Tirar amarelado ou sépia", preprocess.remove_color_cast)
        self.auto_levels = make_checkbox("Recuperar contraste", preprocess.auto_levels)
        self.denoise = make_checkbox("Reduzir ruído", preprocess.denoise)
        self.denoise_strength = make_slider(2.0, preprocess.denoise_strength)
        self.clahe = make_checkbox("Contraste local", preprocess.clahe)
        self.clahe_strength = make_slider(1.0, preprocess.clahe_strength)
        self.sharpen = make_checkbox("Nitidez", preprocess.sharpen)
        self.sharpen_amount = make_slider(1.0, preprocess.sharpen_amount)

        # "&&" é como o Qt escreve um "&" de verdade; um só viraria atalho de teclado e sumiria do texto.
        self.colorize = make_checkbox("Colorir (foto P&&B)", False)
        self.colorize_strength = make_slider(1.0, DEFAULT_STRENGTH)
        if not colorize_available():
            self.colorize.setEnabled(False)
            self.colorize.setText("Colorir (modelo não baixado)")
            self.colorize.setToolTip("Rode: python -m retrato_vivo modelos")
        self.colorize_model = QComboBox()
        self.colorize_model.addItem("DDColor", "ddcolor")
        self.colorize_model.addItem("Zhang", "zhang")
        if not ddcolor.is_available():
            self.colorize_model.setCurrentIndex(1)
        if ddcolor.is_available() and not self.colorize.isEnabled():
            self.colorize.setEnabled(True)
            self.colorize.setText("Colorir (foto P&&B)")
        self.colorize_model.setToolTip("DDColor: mais realista. Zhang: mais discreto. Se as cores saírem "
                                       "estranhas (ex.: daguerreótipo com moldura), troque de modelo.")
        self.restore_faces = make_checkbox("Melhorar rostos", False)
        self.faces_strength = make_slider(1.0, face_restore.DEFAULT_STRENGTH)
        if not face_restore.is_available():
            self.restore_faces.setEnabled(False)
            self.restore_faces.setText("Rostos (sem modelo)")
        self.restore_faces.setToolTip("GFPGAN: reconstrói olhos, boca e pele de rostos borrados.")
        self.upscale = make_checkbox("Ampliar foto pequena", False)
        if not superres.is_available("espcn", 2):
            self.upscale.setEnabled(False)
            self.upscale.setText("Ampliar (sem modelo)")
        self.upscale.setToolTip(f"Super-resolução: amplia fotos pequenas (até {superres.DEFAULT_TARGET} px) "
                                "recuperando nitidez de bordas e texturas. Foto grande passa direto.")
        self.upscale_model = QComboBox()
        self.upscale_model.addItem("ESPCN (rápido)", "espcn")
        self.upscale_model.addItem("EDSR (melhor, lento)", "edsr")
        self.upscale_model.setToolTip("EDSR leva até meio minuto e refaz a cada ajuste nos controles. "
                                      "Use para o resultado final.")
        # ---- Na frente: o que quase todo mundo usa. -----------------------------------
        self.sensitivity.setToolTip("Mais para a direita: apaga mais riscos e manchas (e arrisca apagar "
                                    "detalhes da foto).")
        essentials = QGroupBox("Restauração")
        form = QFormLayout(essentials)
        form.addRow("Força da limpeza", self.sensitivity)
        form.addRow(self.colorize, self.colorize_strength)
        form.addRow(self.restore_faces)
        form.addRow(self.upscale)
        hint = QLabel("Sobrou algum dano? Pinte-o na aba Máscara.")
        hint.setWordWrap(True)
        hint.setEnabled(False)                    # texto em cinza: é dica, não controle
        form.addRow(hint)

        # ---- Atrás de "Ajustes avançados": o resto, fechado por padrão. ------------------
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Ajustes avançados")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced = QWidget()
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        detection = QGroupBox("O que procurar")
        form = QFormLayout(detection)
        for checkbox in (self.use_ai, self.detect_scratches, self.detect_creases, self.detect_spots):
            form.addRow(checkbox)
        form.addRow("Preenchimento", self.inpaint_method)

        corrections = QGroupBox("Correções de imagem")
        form = QFormLayout(corrections)
        form.addRow(self.remove_color_cast)
        form.addRow(self.auto_levels)
        form.addRow(self.denoise, self.denoise_strength)     # a caixa liga a correção, o slider dá a intensidade
        form.addRow(self.clahe, self.clahe_strength)
        form.addRow(self.sharpen, self.sharpen_amount)

        deep_learning = QGroupBox("Inteligência artificial")
        form = QFormLayout(deep_learning)
        form.addRow("Modelo de cor", self.colorize_model)
        form.addRow("Força nos rostos", self.faces_strength)
        form.addRow("Ampliação", self.upscale_model)

        for group in (detection, corrections, deep_learning):
            advanced_layout.addWidget(group)
        self.advanced.setVisible(False)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)

        layout = QVBoxLayout(self)
        layout.addWidget(essentials)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.advanced)
        layout.addStretch()

        # Qualquer mexida avisa a janela, que refaz a restauração sozinha. Procurar os controles
        # pelo tipo faz um controle novo já entrar nesta lista.
        for slider in self.findChildren(QSlider):
            slider.valueChanged.connect(self._announce_change)
        for checkbox in self.findChildren(QCheckBox):
            checkbox.toggled.connect(self._announce_change)
        self.inpaint_method.currentIndexChanged.connect(self._announce_change)
        self.colorize_model.currentIndexChanged.connect(self._announce_change)
        self.upscale_model.currentIndexChanged.connect(self._announce_change)

    def _toggle_advanced(self, show: bool) -> None:
        self.advanced.setVisible(show)
        self.advanced_toggle.setArrowType(Qt.ArrowType.DownArrow if show else Qt.ArrowType.RightArrow)

    def _announce_change(self) -> None:
        self.changed.emit()

    def post_stages(self) -> list[tuple[str, PostStage]]:
        """As etapas de IA ligadas, no formato de ``restore(post_stages=...)``."""
        stages = []
        # Rosto antes da cor: a restauração de rosto trabalha melhor na foto ainda em cinza.
        if self.restore_faces.isChecked():
            stages.append(("rostos_ia", face_restore.make_face_restore_stage(slider_value(self.faces_strength))))
        if self.colorize.isChecked():
            strength = slider_value(self.colorize_strength)
            if self.colorize_model.currentData() == "ddcolor":
                stages.append(("colorizacao", ddcolor.make_ddcolor_stage(strength)))
            else:
                stages.append(("colorizacao", make_colorize_stage(strength)))
        # Ampliação por último: colorir e restaurar rostos na foto menor é mais rápido.
        if self.upscale.isChecked():
            stages.append(("super_resolucao", superres.make_upscale_stage(self.upscale_model.currentData())))
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
        cfg.damage.use_ai = self.use_ai.isChecked()

        cfg.preprocess.remove_color_cast = self.remove_color_cast.isChecked()
        cfg.preprocess.auto_levels = self.auto_levels.isChecked()
        cfg.preprocess.denoise = self.denoise.isChecked()
        cfg.preprocess.denoise_strength = slider_value(self.denoise_strength)
        cfg.preprocess.clahe = self.clahe.isChecked()
        cfg.preprocess.clahe_strength = slider_value(self.clahe_strength)
        cfg.preprocess.sharpen = self.sharpen.isChecked()
        cfg.preprocess.sharpen_amount = slider_value(self.sharpen_amount)
        return cfg
