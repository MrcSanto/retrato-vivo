"""Etapas com Deep Learning (responsável: Marco Antonio Santolin).

Ver docs/FLUXO_DE_TRABALHO.md, seção "Marco". Cada etapa deve seguir o
contrato ``PostStage`` de ``retrato_vivo.pipeline``.
"""

from .colorize import colorize, colorize_image, make_colorize_stage

__all__ = ["colorize", "colorize_image", "make_colorize_stage"]
