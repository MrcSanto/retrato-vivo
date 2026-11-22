"""Etapas com Deep Learning (responsável: Marco Antonio Santolin).

Ver docs/FLUXO_DE_TRABALHO.md, seção "Marco". Cada etapa deve seguir o
contrato ``PostStage`` de ``retrato_vivo.pipeline``.
"""

from .colorize import colorize, colorize_image, make_colorize_stage
from .edges import bordas_canny, bordas_hed, painel_bordas
from .superres import make_upscale_stage, upscale, upscale_image

__all__ = ["colorize", "colorize_image", "make_colorize_stage",
           "upscale", "upscale_image", "make_upscale_stage",
           "bordas_canny", "bordas_hed", "painel_bordas"]
