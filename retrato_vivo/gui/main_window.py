"""Janela principal: junta as abas, os controles e a barra de status."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QProgressBar, QScrollArea, QTabWidget)

from .. import models, runtime
from ..io_utils import SUPPORTED_EXTENSIONS, load_image, save_image
from ..pipeline import RestorationConfig, RestorationResult
from ..visualize import side_by_side
from .compare_view import CompareView
from .controls import ControlsPanel
from .edges_view import EdgesView
from .extras import LivingPortraitView, TimeMachineView
from .folder_nav import PhotoFolder, PhotoPicker
from .mask_editor import MaskEditor
from .steps_view import StepsView
from .worker import RestoreWorker

# "Todos os arquivos" existe porque load_image também abre formatos fora da lista
# (AVIF, HEIC) e fotos salvas com a extensão errada.
OPEN_FILTER = ("Imagens (" + " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
               + ");;Todos os arquivos (*)")
SAVE_FILTER = "PNG (*.png);;JPEG (*.jpg);;Todos os arquivos (*)"
RERUN_DELAY_MS = 400    # espera depois da última mexida nos controles antes de restaurar de novo


FACE_MODEL = "yunet"    # o nome que faces.py usa no registro de modelos


def restoration_summary(result: RestorationResult) -> str:
    """Resumo para a barra de status: os danos encontrados e o tempo total."""
    seconds = sum(result.timings_ms.values()) / 1000
    time_note = f"tempo: {_decimal(seconds)} s"
    if result.damage is None:      # reparo de danos desligado na configuração
        return time_note
    stats = result.damage.stats
    return " · ".join([
        f"riscos: {stats['riscos']:.0f}",
        f"vincos: {stats['vincos']:.0f}",
        f"manchas: {stats['manchas']:.0f}",
        f"área danificada: {_decimal(stats['area_danificada_pct'])}%",
        f"rostos protegidos: {stats['rostos_protegidos']:.0f}",
        time_note,
    ])


def _decimal(value: float) -> str:
    """Número com duas casas e vírgula decimal, como se escreve em português."""
    return f"{value:.2f}".replace(".", ",")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Retrato Vivo")
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        self.image: np.ndarray | None = None            # a foto aberta, como load_image devolveu
        self.photo_path: Path | None = None
        self.result: RestorationResult | None = None    # a última restauração da foto aberta
        self._result_config: RestorationConfig | None = None    # a configuração que gerou ``result``
        self.worker: RestoreWorker | None = None
        self.busy = False                               # há uma restauração rodando?

        # Refaz a restauração quando o usuário para de mexer nos controles. "Tiro único": cada
        # mexida recomeça a contagem, então arrastar um slider gera uma restauração só, no final.
        self.rerun_timer = QTimer(self)
        self.rerun_timer.setSingleShot(True)
        self.rerun_timer.setInterval(RERUN_DELAY_MS)
        self.rerun_timer.timeout.connect(self._restore_after_changes)

        self.toolbar = self.addToolBar("Principal")
        self.toolbar.setMovable(False)
        open_action = self.toolbar.addAction("Abrir", self.choose_photo)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.restore_action = self.toolbar.addAction("Restaurar", self.restore_photo)
        self.restore_action.setEnabled(False)           # só faz sentido com uma foto aberta
        self.save_action = self.toolbar.addAction("Salvar resultado", self.save_result)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.export_action = self.toolbar.addAction("Exportar comparativo", self.export_comparison)
        self._enable_saving(False)                      # só há o que salvar depois de uma restauração

        # Navegação pelas fotos da pasta da foto aberta (folder_nav.py).
        self.photo_folder = PhotoFolder(self)
        self.toolbar.addSeparator()
        self.previous_action = self.toolbar.addAction("◀", lambda: self.open_neighbor(-1))
        self.previous_action.setShortcut(QKeySequence.StandardKey.MoveToPreviousPage)    # PgUp
        self.previous_action.setToolTip("Foto anterior da pasta (PgUp)")
        self.photo_picker = PhotoPicker(self.photo_folder)
        self.photo_picker.activated.connect(self._open_picked)
        self.toolbar.addWidget(self.photo_picker)
        self.next_action = self.toolbar.addAction("▶", lambda: self.open_neighbor(+1))
        self.next_action.setShortcut(QKeySequence.StandardKey.MoveToNextPage)            # PgDn
        self.next_action.setToolTip("Próxima foto da pasta (PgDn)")
        self.photo_folder.changed.connect(self._update_folder_navigation)
        self._update_folder_navigation()

        self.compare_view = CompareView()
        self.compare_view.placeholder = "Arraste uma foto para cá ou clique em Abrir"
        self.tabs = QTabWidget()
        self.tabs.addTab(self.compare_view, "Antes e depois")
        self.steps_view = StepsView()
        self.tabs.addTab(self.steps_view, "Etapas")
        self.mask_editor = MaskEditor()
        self.tabs.addTab(self.mask_editor, "Máscara")
        self.mask_editor.apply_button.clicked.connect(self.restore_photo)
        self.edges_view = EdgesView()
        self.tabs.addTab(self.edges_view, "Bordas")
        # Diferencial do projeto (extras.py).
        self.living_view = LivingPortraitView()
        self.tabs.addTab(self.living_view, "Retrato Vivo")
        self.time_machine = TimeMachineView()
        self.tabs.addTab(self.time_machine, "Máquina do tempo")
        self.setCentralWidget(self.tabs)

        self.controls = ControlsPanel()
        self.controls.changed.connect(self.rerun_timer.start)
        scroll = QScrollArea()                         # em tela baixa o painel rola em vez de ser cortado
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        # Largura mínima: a do painel, mais a barra de rolagem vertical e a moldura. Sem isto,
        # uma aba larga espreme o painel e ele passa a rolar para o lado.
        scroll.setMinimumWidth(self.controls.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
                               + 2 * scroll.frameWidth())
        dock = QDockWidget("Controles")
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)    # fixo: não fecha nem destaca
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)                    # sem faixa, a barra só anima: "ocupado"
        self.progress.setMaximumWidth(160)
        self.progress.hide()
        if not models.is_available(FACE_MODEL):
            # Sem o modelo o pipeline funciona, mas não protege os rostos na detecção de danos.
            self.statusBar().addPermanentWidget(
                QLabel("Detector de rostos não baixado. Rode: python -m retrato_vivo modelos"))
        self.statusBar().addPermanentWidget(self.progress)
        # Onde as redes pesadas rodam: na apresentação mostra se a GPU está sendo usada.
        self.device_label = QLabel(f"IA: {runtime.device_name()}" if runtime.is_available() else "IA: indisponível")
        self.device_label.setToolTip("GPU (CUDA) quando há placa NVIDIA; senão CPU (funciona igual, mais devagar).")
        self.statusBar().addPermanentWidget(self.device_label)
        self.statusBar().showMessage("Abra uma foto ou arraste o arquivo para a janela.")

    # ------------------------------------------------------------ abrir a foto

    def choose_photo(self) -> None:
        """Pergunta qual arquivo abrir. Cancelar o diálogo não muda nada."""
        path, _ = QFileDialog.getOpenFileName(self, "Abrir foto", "", OPEN_FILTER)
        if path:
            self.open_photo(path)

    def open_photo(self, path: str | Path) -> None:
        """Abre a foto, mostra e já manda restaurar. Se o arquivo não abrir, avisa e mantém a foto atual."""
        path = Path(path)
        try:
            image = load_image(path)
        except (ValueError, OSError) as error:     # formato desconhecido, arquivo ausente ou sem permissão
            QMessageBox.warning(self, "Não foi possível abrir a foto", str(error))
            return
        self.image, self.photo_path, self.result = image, path, None
        self.compare_view.set_photo(image)
        self.steps_view.clear()
        self.mask_editor.clear()             # a pintura era da foto anterior
        self.edges_view.set_result(None)
        self.living_view.set_source(None)    # o Retrato Vivo usa a foto restaurada, que ainda não existe
        self.time_machine.set_source(image)
        self._enable_saving(False)           # o resultado antigo era de outra foto
        self.restore_action.setEnabled(True)
        self.photo_folder.follow(path)
        self.restore_photo()

    def open_neighbor(self, step: int) -> None:
        """Abre a foto ``step`` posições adiante na pasta (negativo = para trás)."""
        photo = self.photo_folder.neighbor(self.photo_path, step)
        if photo is not None and not self.busy:
            self.open_photo(photo)

    def _open_picked(self, index: int) -> None:
        photo = Path(self.photo_picker.itemData(index))
        if photo != (self.photo_path.resolve() if self.photo_path else None):
            self.open_photo(photo)

    def _update_folder_navigation(self) -> None:
        self.photo_picker.show_photos(self.photo_path)
        can_move = len(self.photo_folder.photos) > 1 or (
            bool(self.photo_folder.photos) and self.photo_folder.index_of(self.photo_path) < 0)
        self.previous_action.setEnabled(can_move)
        self.next_action.setEnabled(can_move)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """O Qt só entrega o ``dropEvent`` se o arrasto for aceito aqui."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if files:
            self.open_photo(files[0])

    # ------------------------------------------------------------------ salvar

    def save_result(self) -> None:
        self._save_as("Salvar resultado", "_restaurada.png", self.result.image)

    def export_comparison(self) -> None:
        comparison = side_by_side(self.result.original, self.result.image)
        self._save_as("Exportar comparativo", "_antes_depois.jpg", comparison)

    def _save_as(self, title: str, suffix: str, image: np.ndarray) -> None:
        """Pergunta onde gravar, sugerindo o nome da foto aberta mais ``suffix``, e grava ``image``."""
        suggestion = self.photo_path.with_name(self.photo_path.stem + suffix)
        path, _ = QFileDialog.getSaveFileName(self, title, str(suggestion), SAVE_FILTER)
        if not path:      # o usuário cancelou
            return
        try:
            saved = save_image(path, image)
        # cv2.error e ValueError: extensão que o OpenCV não grava. OSError: pasta inválida ou sem permissão.
        except (cv2.error, ValueError, OSError) as error:
            QMessageBox.warning(self, "Não foi possível salvar",
                                f"Não foi possível salvar em {path}.\n\n{error}\n\n"
                                "Confira a pasta e a extensão (.png ou .jpg).")
            return
        self.statusBar().showMessage(f"Salvo em {saved}")

    def _enable_saving(self, enabled: bool) -> None:
        self.save_action.setEnabled(enabled)
        self.export_action.setEnabled(enabled)

    # --------------------------------------------------------------- restaurar

    def restore_photo(self) -> None:
        """Restaura a foto aberta em segundo plano. O resultado chega em ``_show_result``."""
        self._wait_for_worker()
        config = self.controls.config()
        mask_add, mask_remove = self._masks_for(config)
        self.worker = RestoreWorker(self.image, config,
                                    post_stages=self.controls.post_stages(),
                                    mask_add=mask_add, mask_remove=mask_remove)
        self.worker.done.connect(self._show_result)
        self.worker.failed.connect(self._show_failure)
        self._set_busy(True)
        self.statusBar().showMessage(f"{self._photo_label()} · restaurando…")
        self.worker.start()

    def _masks_for(self, config: RestorationConfig) -> tuple[np.ndarray | None, np.ndarray | None]:
        """As máscaras pintadas, se ainda servem para a foto que ``config`` vai gerar.

        A correção de perspectiva recorta a foto. A pintura feita num enquadramento não vale
        para o outro e nem tem o mesmo tamanho: o pipeline falharia ao aplicá-la.
        """
        if self.result is not None and config.correct_perspective != self._result_config.correct_perspective:
            self.mask_editor.canvas.clear_masks()
            return None, None
        return self.mask_editor.canvas.masks()

    def _restore_after_changes(self) -> None:
        """Chamado pelo ``rerun_timer`` quando o usuário para de mexer nos controles."""
        if self.image is None:      # sem foto aberta não há o que refazer
            return
        if self.busy:
            self.rerun_timer.start()    # já há uma restauração rodando: tenta de novo daqui a pouco
            return
        self.restore_photo()

    def _show_result(self, result: RestorationResult) -> None:
        self.result, self._result_config = result, self.worker.config
        self.compare_view.set_comparison(result.original, result.image)
        self.steps_view.set_result(result)
        self.mask_editor.set_result(result)
        self.edges_view.set_result(result)
        self.living_view.set_source(result.image)
        self._enable_saving(True)
        self._set_busy(False)
        self.statusBar().showMessage(f"{self._photo_label()} · {restoration_summary(result)}")

    def _show_failure(self, message: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage(f"{self._photo_label()} · a restauração falhou")
        QMessageBox.warning(self, "Não foi possível restaurar a foto", message)

    def _set_busy(self, busy: bool) -> None:
        """Enquanto uma restauração roda, a janela não aceita outra foto nem outro pedido.

        O pipeline guarda os modelos em variáveis globais, sem trava: duas
        restaurações ao mesmo tempo poderiam se atrapalhar.
        """
        self.busy = busy
        self.toolbar.setEnabled(not busy)
        self.mask_editor.apply_button.setEnabled(not busy)
        self.setAcceptDrops(not busy)
        self.progress.setVisible(busy)

    def _wait_for_worker(self) -> None:
        """O Qt aborta o programa se um ``QThread`` for destruído enquanto ainda roda."""
        if self.worker is not None:
            self.worker.wait()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.rerun_timer.stop()      # nada de começar outra restauração com a janela fechando
        self._wait_for_worker()
        self.living_view.shutdown()
        self.time_machine.shutdown()
        self.edges_view.shutdown()
        super().closeEvent(event)

    def _photo_label(self) -> str:
        height, width = self.image.shape[:2]
        return f"{self.photo_path.name} · {width} × {height} px"
