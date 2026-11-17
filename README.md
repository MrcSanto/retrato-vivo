# Retrato Vivo

Restaurador de fotos antigas com OpenCV e Deep Learning. Trabalho final de
**CCC309 – Processamento de Imagens e Visão Computacional** (UPF, 2026/2).

**Equipe:** Igor Zanette · Marco Antonio Santolin · Marcos Paulo de Medeiros

🎬 **Vídeo demonstrativo:** _(em breve)_

## O que faz

- **Mede** a foto (cor, ruído, contraste) e só aplica as correções de que ela precisa.
- Detecta **riscos, vincos e manchas** (morfologia matemática) e os apaga com
  **inpainting**, protegendo os rostos com um detector neural (YuNet).
- Remove amarelado/sépia, recupera o contraste de fotos desbotadas, reduz a granulação
  (Non-Local Means) e realça detalhes (CLAHE + nitidez).
- Explica o que fez: mapa de danos, painel com todas as etapas e o motivo de cada decisão.
- _Em desenvolvimento:_ colorização e super-resolução com Deep Learning, interface
  gráfica, **Retrato Vivo** (a foto restaurada ganha movimento 3D) e **Máquina do tempo**
  (uma foto atual vira foto de outra época).

## Como usar

```bash
pip install -r requirements.txt
python -m retrato_vivo modelos                       # baixa os modelos pré-treinados
python -m retrato_vivo.gui [foto.jpg]                # interface gráfica (em construção)
python -m retrato_vivo restaurar foto.jpg -o saida --painel --explicar
python -m retrato_vivo avaliar foto_boa.jpg          # estraga, restaura e mede PSNR/SSIM
python -m pytest
```

## Documentação

- **[docs/COMO_FUNCIONA.md](docs/COMO_FUNCIONA.md)**: o que cada arquivo e cada etapa faz.
- **[docs/FLUXO_DE_TRABALHO.md](docs/FLUXO_DE_TRABALHO.md)**: divisão de tarefas, prazos e regras para contribuir.
