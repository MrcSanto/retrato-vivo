# O diferencial: Retrato Vivo e Máquina do tempo

O professor pediu que o sistema faça "algo a mais" além de restaurar. A equipe escolheu:

| | O que faz | Onde |
|---|---|---|
| ⭐ **Retrato Vivo** | A foto restaurada vira um vídeo curto com efeito 3D: uma câmera virtual passeia diante dela e a pessoa se destaca do fundo. Dá nome ao projeto. | aba **Retrato Vivo**; `python -m retrato_vivo vivo foto.jpg`; [`living_portrait.py`](../retrato_vivo/living_portrait.py) |
| Bônus: **Máquina do tempo** | O inverso da restauração: uma foto atual vira foto de 1900, 1920, 1950, 1970 ou Polaroid dos anos 90. "Restaurar de volta" mostra a restauração desfazendo o envelhecimento. | aba **Máquina do tempo**; `python -m retrato_vivo viajar foto.jpg`; [`time_machine.py`](../retrato_vivo/time_machine.py) |

## Retrato Vivo: como funciona

1. **Profundidade (Deep Learning).** O Depth Anything V2 Small (Yang et al., NeurIPS 2024,
   Apache-2.0) estima, para cada pixel, o quão perto ele está da câmera. Com uma foto só, sem
   câmera 3D. Na interface, "Ver profundidade" mostra o mapa (claro = perto).
2. **Camadas.** O limiar de Otsu sobre a profundidade separa o primeiro plano (a pessoa) do fundo;
   a morfologia fecha buracos e remove ilhas, e um desfoque suaviza a borda.
3. **Fundo escondido (Deep Learning).** Quando a câmera se move, aparece o que estava atrás da
   pessoa. O LaMa apaga a pessoa e inventa esse pedaço de fundo antes (por exemplo, reconstrói a
   parede e a porta atrás do homem da `foto3`).
4. **Câmera virtual.** Em cada quadro a câmera percorre uma elipse e aproxima levemente. Cada
   pixel é deslocado com `cv2.remap` proporcionalmente à sua profundidade (paralaxe: o perto se
   mexe mais), e a pessoa é composta sobre o fundo com a borda suave.
5. **Saída.** MP4 (`cv2.VideoWriter`) e GIF. 4 s, 25 quadros por segundo, repetição sem emenda.

Tempo: 2 a 4 s na GPU (RTX 4080); na CPU, alguns segundos a mais.

**Limitações:** funciona melhor em retratos e cenas com um primeiro plano claro. Em foto sem
pessoa (paisagem plana) o efeito é discreto. Cabelo fino na borda pode "descolar" um pouco.

## Máquina do tempo: como funciona

Cada época é uma sequência dos efeitos de [`aging.py`](../retrato_vivo/aging.py), a mesma
biblioteca usada para avaliar a restauração:

| Época | Receita |
|---|---|
| 1900, retrato de estúdio | P&B, lente suave, sépia forte, desbotamento, vinheta forte, grão, muitos riscos e poeira, borda de papel |
| 1920, álbum de família | P&B, sépia média, desbotamento, vinheta, grão, riscos e poeira, borda |
| 1950, fotojornalismo | P&B contrastado, vinheta leve, grão fino, pouca poeira, borda branca |
| 1970, cópia colorida | cores fracas, dominante amarelo-avermelhada, pretos lavados, grão |
| 1990, Polaroid | quadrada, contraste suave, sombras esverdeadas, data laranja, moldura de base larga |

**Ida e volta:** a foto envelhecida passa pela restauração completa. É uma demonstração ao vivo
de que a detecção e o preenchimento funcionam: os riscos que a Máquina do tempo desenhou somem.

## Roteiro sugerido para a apresentação

1. Abrir a `foto_risco3` → mostrar o antes/depois (riscos somem, rosto melhora, cor).
2. Aba **Etapas** → explicar o processamento clássico + IA.
3. Aba **Retrato Vivo** → "Criar" → a foto ganha vida. Mostrar "Ver profundidade".
4. Pedir uma selfie ao professor → aba **Máquina do tempo** → 1920 → "Restaurar de volta".
