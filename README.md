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
- **Colore** fotos preto e branco com uma rede neural (Zhang et al., ECCV 2016).
- Tem **interface gráfica**: comparação antes e depois, a foto em cada etapa, ajustes da
  restauração e um editor para corrigir a detecção à mão.
- _Em desenvolvimento:_ super-resolução com Deep Learning, **Retrato Vivo** (a foto
  restaurada ganha movimento 3D) e **Máquina do tempo** (uma foto atual vira foto de
  outra época).

## Como usar

```bash
pip install -r requirements.txt
python -m retrato_vivo modelos                       # baixa os modelos pré-treinados
python -m retrato_vivo.gui [foto.jpg]                # interface gráfica
python -m retrato_vivo restaurar foto.jpg -o saida --painel --explicar
python -m retrato_vivo avaliar foto_boa.jpg          # estraga, restaura e mede PSNR/SSIM
python -m pytest
```

## Interface gráfica

Abra com `python -m retrato_vivo.gui`. Para já começar com uma foto, passe o arquivo:
`python -m retrato_vivo.gui foto.jpg`.

1. **Abrir:** clique em "Abrir" (Ctrl+O) ou arraste o arquivo para a janela. A restauração
   começa sozinha e não trava a janela.
2. **Antes e depois:** arraste a linha branca sobre a foto. A original fica à esquerda
   ("Antes") e a restaurada à direita ("Depois").
3. **Etapas:** clique em uma etapa para ver a foto naquele ponto, o que a etapa faz e o que
   ela decidiu nessa foto. O rodapé mostra as etapas puladas e o tempo de cada fase.
4. **Controles:** no painel da esquerda, mude a sensibilidade, ligue ou desligue correções
   e marque "Colorir foto P&B". A restauração é refeita sozinha assim que você para de
   mexer. O botão "Restaurar" refaz na hora.
5. **Máscara:** com o "Pincel", pinte o dano que a detecção não pegou. Com a "Borracha",
   desmarque o que foi marcado por engano. "Desfazer" (Ctrl+Z) volta um traço. Gire a roda
   do mouse para ampliar a foto e arraste com o botão direito para movê-la. O botão "Zoom"
   volta à foto inteira. "Aplicar correções" refaz a restauração com o que foi pintado.
6. **Salvar:** "Salvar resultado" (Ctrl+S) grava a foto restaurada. "Exportar comparativo"
   grava o antes e o depois lado a lado.

A barra de status mostra quantos riscos, vincos e manchas foram encontrados, a área
danificada, os rostos protegidos e o tempo total.

## Documentação

- **[docs/COMO_FUNCIONA.md](docs/COMO_FUNCIONA.md)**: o que cada arquivo e cada etapa faz.
- **[docs/FLUXO_DE_TRABALHO.md](docs/FLUXO_DE_TRABALHO.md)**: divisão de tarefas, prazos e regras para contribuir.
