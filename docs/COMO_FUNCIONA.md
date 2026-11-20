# Como funciona a restauração

Guia da parte do Igor: **pré-processamento, detecção e preenchimento de danos**.
A base é processamento de imagens clássico com OpenCV. Desde 20/11, duas redes neurais
reforçam os pontos onde as regras não davam conta: um **detector de riscos** treinado em
fotos antigas reais e o **LaMa**, que preenche áreas grandes. Elas rodam na GPU (CUDA)
quando houver e na CPU nos outros PCs (seção 9).

Em uma frase: o sistema **mede** a foto, corrige cor e contraste só se precisar,
**marca** onde há riscos, vincos e manchas, **apaga** o que marcou e dá o acabamento.

## 1. O que cada arquivo faz

| Arquivo | Pergunta que ele responde |
|---|---|
| [`io_utils.py`](../retrato_vivo/io_utils.py) | Como abrir e salvar qualquer foto (acentos, celular deitado, PNG transparente, 16 bits)? |
| [`preprocess.py`](../retrato_vivo/preprocess.py) | A foto está amarelada, desbotada, granulada? Como corrigir sem estragar? |
| [`faces.py`](../retrato_vivo/faces.py) | Onde estão os rostos (para protegê-los)? |
| [`damage.py`](../retrato_vivo/damage.py) | **Onde** estão os riscos, vincos e manchas? (regras) |
| [`damage_ai.py`](../retrato_vivo/damage_ai.py) | Quais riscos uma **rede treinada em fotos antigas** acha? (soma com as regras) |
| [`inpaint.py`](../retrato_vivo/inpaint.py) | Como **apagar** o que foi marcado? (Telea nos riscos finos, LaMa nas áreas grandes) |
| [`runtime.py`](../retrato_vivo/runtime.py) | As redes pesadas rodam na GPU ou na CPU? |
| [`visualize.py`](../retrato_vivo/visualize.py) | Como mostrar o que foi feito (mapa de danos, antes/depois, painel)? |
| [`pipeline.py`](../retrato_vivo/pipeline.py) | Em que ordem tudo roda? |
| [`aging.py`](../retrato_vivo/aging.py) | Como estragar uma foto boa de propósito, para testar? (os mesmos efeitos montam a Máquina do tempo) |
| [`metrics.py`](../retrato_vivo/metrics.py) | Quão boa ficou a restauração, em números? |
| [`cli.py`](../retrato_vivo/cli.py) | Como rodar tudo pelo terminal? |

Ordem sugerida de leitura do código: `pipeline.py` → `preprocess.py` → `damage.py` → `inpaint.py`.

## 2. O caminho de uma foto

A regra de ouro é **não estragar foto boa**: cada etapa mede se a foto precisa dela.
Quando não precisa, a etapa é pulada e o motivo fica registrado.

| # | Etapa | O que faz | Técnica | Quando age |
|---|---|---|---|---|
| 1 | Abrir | Lê o arquivo e reduz para no máximo 2048 px | `imdecode`, rotação EXIF | Sempre |
| 2 | Diagnóstico | Mede cor, ruído e contraste | Estatísticas em LAB, filtro de ruído | Sempre (não altera nada) |
| 3 | Cor | Foto sépia/P&B vira cinza neutro; foto colorida amarelada perde o amarelo | Espaço LAB | Só se houver dominante de cor |
| 4 | Contraste | Devolve o preto e o branco a uma foto desbotada | Esticamento de histograma | Só se a foto não usa toda a faixa de tons |
| 5 | Detectar danos | Marca riscos, vincos e manchas numa máscara | Morfologia matemática, componentes conexos **+ U-Net (IA)** | Sempre |
| 6 | Apagar danos | Preenche a máscara | Telea nos riscos finos; **LaMa (IA)** nos buracos largos | Onde a máscara marcou |
| 7 | Ruído | Reduz a granulação | Non-Local Means | Só se o ruído medido for alto; a força acompanha o ruído |
| 8 | Contraste local | Recupera detalhe em áreas escuras e claras | CLAHE misturado com o original | Sempre, em dose leve (40%) |
| 9 | Nitidez | Realça contornos | Máscara de desfoque | Sempre |

Por que nessa ordem: cor e contraste vêm **antes** da detecção para que ela funcione
igual em qualquer foto. Ruído e nitidez vêm **depois** do reparo, porque a nitidez
realçaria justamente os riscos.

### Como cada medida do diagnóstico é feita

- **É preto e branco/sépia?** Dois testes; basta um passar:
  1. A cor quase não *varia* entre os pixels (desvio padrão de a, b do LAB menor que 6).
     Pega P&B e sépia suave: o sépia desloca a cor, mas varia pouco.
  2. A cor varia, mas é **sempre o mesmo tom**: menos de 5% da cor fica fora de uma faixa
     de ±30° em torno do tom dominante. Pega sépia de alto contraste, em que as sombras
     são marrom escuro e os claros são creme. É como café com leite: marrom escuro e bege
     são o mesmo tom em quantidades diferentes. Uma foto colorida tem vários tons (céu,
     pele, folhas) e fica bem acima de 10%.
- **Tem dominante de cor?** Toda foto sem dominante tem algum pixel neutro (um branco,
  um cinza). Se nem o 1% de pixels mais neutros é neutro, a cor deles é a dominante.
- **Quanto ruído?** Um filtro cancela as regiões lisas e sobra quase só ruído; a
  mediana do resultado ignora as bordas da cena. Foto digital limpa dá menos de 1,
  filme granulado dá de 3 a 10.

## 3. Como a detecção separa dano de cena

Este é o coração do trabalho. Os três detectores seguem a mesma receita:

1. **Realçar** tudo o que *parece* o dano.
2. **Agrupar** os pixels realçados em regiões candidatas.
3. **Filtrar**: cada candidata passa por testes com nome. Quem falha é descartada.

| Detector | O que é | Como realça |
|---|---|---|
| **Riscos** | Arranhão fino, claro ou escuro | Top-hat (fino e claro) + black-hat (fino e escuro) + abertura com segmentos de reta em 12 inclinações (reto e longo) |
| **Vincos** | Dobra ou rachadura do papel: linha branca mais larga | A mesma receita dos riscos, com largura maior e só linhas claras |
| **Manchas** | Poeira: ponto pequeno, claro ou escuro | Diferença entre a foto e a mediana local (a mediana apaga tudo que é pequeno) |

Os testes do passo 3 e o que cada um evita:

| Teste | Descarta a candidata quando... | Exemplo do que deixa de ser marcado |
|---|---|---|
| textura (antes de agrupar) | há muitas coisas finas na vizinhança | cabelo, plumas, tecido |
| curto demais | é menor que 2,5% do maior lado | cílios, pequenos detalhes |
| pouco alongado | não é comprida e fina | letras, cantos |
| não é branco *(só vincos)* | está mais de 60 níveis abaixo do ponto mais claro da foto | gola, friso de carro |
| em borda da cena | fica sobre uma transição claro/escuro da cena | dobra de vestido, contorno de objeto |
| tamanho errado *(manchas)* | é grande demais para ser poeira | botões, olhos |
| não é compacta *(manchas)* | é esticada ou irregular | traços |
| entorno com detalhe *(manchas)* | o anel em volta dela não é liso | pupila, textura |
| padrão repetido *(manchas)* | tem mais de 2 vizinhas parecidas | janelas de prédio, pedras de muro |

Dentro de **rostos** (detectados pela rede YuNet), os testes ficam mais rígidos, para
não apagar pupilas, cílios ou lábios.

O que sobra entra na máscara, com uma pequena margem ao redor. O usuário ainda pode
**pintar** o que faltou e **apagar** o que foi marcado por engano (`mask_add`,
`mask_remove`), e é isso que o editor de máscara da GUI vai usar.

## 4. Como ver tudo isso funcionando

```bash
python -m retrato_vivo restaurar imagens/foto_risco3.webp -o saida --painel --explicar
```

- `--painel` salva uma imagem única com todas as etapas, cada uma com seu nome.
- `--explicar` mostra o diagnóstico, o que cada etapa decidiu e quantas candidatas
  cada teste descartou:

```
foto_risco3.webp: 177 ms, 13 riscos, 6 vincos, 77 manchas, 2.34% da área
  diagnóstico:
    tipo            preto e branco / sépia
    dominante de cor 1.1
    ruído           0.5
    tons            de 34 a 219 (0 a 255)
  etapas:
    cor_neutralizada     foto P&B/sépia: convertida para cinza neutro
    niveis_automaticos   contraste expandido 1.28x
    mapa_de_danos        13 riscos, 6 vincos, 77 manchas, 2.34% da área
    ruido_removido       pulada: ruído medido 0.5: foto limpa
    clahe                contraste local realçado (40%)
    nitidez              nitidez 50%
  detecção de danos (candidatas descartadas por teste):
    riscos   candidatos: 35, curto demais: 16, pouco alongado: 0, em borda da cena: 6, aceitos: 13
    vincos   candidatos: 34, curto demais: 13, não é branco: 14, pouco alongado: 1, em borda da cena: 0, aceitos: 6
    manchas  candidatos: 364, tamanho errado: 49, não é compacta: 46, entorno com detalhe: 54, em borda da cena: 68, padrão repetido: 70, aceitos: 77
```

No mapa de danos: **vermelho** = risco, **laranja** = vinco, **amarelo** = mancha,
**verde** = rosto protegido.

## 5. Controles que a interface pode oferecer

| Controle | Campo | Efeito |
|---|---|---|
| Sensibilidade (0 a 1) | `cfg.damage.sensitivity` | Mais alto pega mais dano e mais falso positivo |
| Riscos com IA | `cfg.damage.use_ai` | Soma o detector de riscos com IA (padrão: ligado) |
| Ligar/desligar riscos, vincos, manchas | `cfg.damage.detect_*` | |
| Método de preenchimento | `cfg.inpaint_method` | `ia` (padrão: LaMa nos buracos largos), `telea` ou `ns` |
| Força do ruído | `cfg.preprocess.denoise_strength` | 1 = automático; acima de 1 limpa mesmo foto sem ruído |
| Contraste local | `cfg.preprocess.clahe_strength` | 0 a 1 |
| Nitidez | `cfg.preprocess.sharpen_amount` | |
| Ligar/desligar cada correção | `cfg.preprocess.remove_color_cast`, `auto_levels`, `denoise`, `clahe`, `sharpen` | |

## 6. Como medimos a qualidade

Não dá para medir numa foto antiga real, porque não existe a "resposta certa". Então
fazemos o contrário: pegamos uma foto boa, desenhamos riscos e manchas nela
(`aging.py`), restauramos e comparamos com a original.

```bash
python -m retrato_vivo avaliar imagens/foto_boa1.jpg -o avaliacao
```

Resultados com a sensibilidade padrão (0,5):

| Teste | Resultado |
|---|---|
| Detecção em 2 fotos de exemplo do OpenCV (512 px, 4 sorteios): precisão / revocação / F1 | 0,82 / 0,68 / 0,74 |
| Ganho de PSNR depois do reparo, nas mesmas fotos | +4,4 dB |
| `foto_boa1`, `2`, `3`: PSNR danificada → reparada | 22,6 → 30,7 · 23,5 → 27,7 · 22,5 → 29,2 dB |
| `foto_boa1`, `2`, `3`: detecção F1 | 0,69 · 0,51 · 0,69 |
| Vincos marcados por engano nas 3 fotos boas | 0 |
| Tempo por foto | 0,1 a 0,6 s |

*Precisão*: do que foi marcado, quanto era dano. *Revocação*: do dano que existia,
quanto foi marcado. *PSNR*: quanto maior, mais parecida com a original.

Testes automáticos: `python -m pytest` (38 testes da restauração, um bloco por arquivo).

## 7. Limitações conhecidas

- **Dano dentro de textura** (cabelo, plumas) é ignorado de propósito; um vinco que
  cruza o cabelo fica com um trecho sem reparo. Solução: pincel do editor de máscara.
- **Rasgos não são achados sozinhos.** O papel exposto de um rasgo parece um fundo
  claro qualquer: uma regra "claro e liso" marcaria 31% do fundo de estúdio da `foto1`
  e 27% do céu da `foto_boa2`. O fluxo é: **pintar o rasgo com o pincel** e deixar o LaMa
  preencher. O detector com IA pega as rachaduras em volta, não o papel exposto.
- **Foto coberta de arranhões** (`foto_risco1`): mesmo com a IA, sobra muito risco.
- **Fotos muito detalhadas** (fachadas, multidões) ainda geram manchas falsas, em
  torno de 1% da área.
- **Foto colorida sem nenhum pixel neutro** (ex.: `foto_boa3`) pode ter a cor
  levemente esfriada pela correção de dominante.
- **Foto colorida que desbotou toda para um tom só** (cópias dos anos 70, por exemplo) pode
  ser tratada como sépia e recolorida pela rede. Aconteceu em 2 de 7 simulações; nenhuma
  das fotos coloridas reais do projeto foi afetada.
- **AVIF** exige `pillow>=11.3`; com Pillow antigo o arquivo é ignorado com aviso.

## 8. Perguntas que o professor pode fazer

- **Por que morfologia e não Canny/Hough para os riscos?** Canny acha *bordas*
  (transição entre duas regiões); um risco é uma *crista* fina. O top-hat isola
  exatamente o que é mais fino que o elemento estruturante.
- **Por que LAB?** Separa a luminosidade (L) da cor (a, b). Dá para mexer no
  contraste sem alterar a cor, e vice-versa.
- **Por que Non-Local Means e não Gaussiano?** O Gaussiano borra as bordas junto com
  o ruído. O NLM faz a média só entre trechos parecidos da imagem.
- **Por que CLAHE e não equalização de histograma comum?** A comum é global e estoura
  o ruído em áreas lisas. O CLAHE age por blocos e limita o ganho.
- **Onde entra o Deep Learning nesta parte?** Em três redes: o YuNet localiza os rostos;
  a U-Net da Microsoft (*Bringing Old Photos Back to Life*, CVPR 2020) marca os riscos; o
  LaMa (WACV 2022) reconstrói áreas grandes. As regras clássicas continuam: são rápidas,
  explicáveis e pegam manchas e vincos que a rede não foi treinada para achar.
- **Por que não usar só a IA?** As duas se complementam. Na `foto_risco3`, a rede achou
  57 riscos que as regras perdiam; as regras acharam as manchas de poeira e protegem os
  rostos. E a IA pura não explica por que marcou algo; as regras sim (`--explicar`).
- **Vocês treinaram a rede?** Não: usamos os pesos oficiais (licença MIT) e os convertemos
  de PyTorch para ONNX (`scripts/exportar_detector_riscos.py`), conferindo que a saída é a
  mesma (diferença máxima de 5×10⁻⁷).

## 9. IA na GPU ou na CPU

As redes pesadas rodam no ONNX Runtime ([`runtime.py`](../retrato_vivo/runtime.py)):
GPU NVIDIA pelo CUDA quando houver, CPU nos outros PCs, com o mesmo resultado (testado em
`tests/test_ia.py`). A interface mostra onde a IA está rodando na barra de status.

| Restauração completa (detector + LaMa + rosto + DDColor) | Tempo por foto |
|---|---|
| RTX 4080 SUPER (CUDA) | cerca de 0,4 s |
| Só CPU | cerca de 4 s (8 s na primeira foto, que carrega os modelos) |

Para usar a GPU: `pip install "onnxruntime-gpu[cuda,cudnn]"`. Para simular um PC sem placa:
`set RETRATO_VIVO_CPU=1` (Windows) antes de rodar.

| Modelo | Arquivo | Tamanho | Uso |
|---|---|---|---|
| Detector de riscos (Microsoft, MIT) | `scratch_detector.onnx` | 151 MB | `damage_ai.py` |
| LaMa (Apache-2.0) | `lama_fp32.onnx` | 208 MB | `inpaint.py` |
| GFPGAN v1.4 (Apache-2.0) | `gfpgan_1.4.onnx` | 340 MB | `ai/face_restore.py` |
| DDColor (Apache-2.0) | `ddcolor.onnx` | 980 MB | `ai/ddcolor.py` |

Todos ficam em `models/` (fora do git). Baixe com `python -m retrato_vivo modelos`. O
detector de riscos é convertido por nós; enquanto não for publicado num Release do
repositório, gere com o script ou copie o arquivo de quem já tem.
- **Como sabem que funciona?** Pelo teste de envelhecer e restaurar (seção 6) e pelo
  teste de "não estragar foto boa".
