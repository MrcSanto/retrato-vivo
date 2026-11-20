# Retrato Vivo: fluxo de trabalho da equipe

> Documento vivo. **Leia inteiro antes de começar a sua parte** e atualize a
> seção "Status" da sua área quando terminar algo.

| | |
|---|---|
| **Disciplina** | CCC309 – Processamento de Imagens e Visão Computacional (UPF, 2026/2) |
| **Professor** | Dr. Rafael Rieder (GitHub: `rafaelrieder`) |
| **Tema** | Restaurador de fotos antigas, com dois extras (ver [seção 7](#7-o-algo-a-mais)) |
| **Equipe** | Igor Zanette · Marco Antonio Santolin · Marcos Paulo de Medeiros |
| **Stack** | Python 3.12+ · OpenCV 4.x (`cv2`, `cv2.dnn`; o 5 não lê Caffe) · NumPy · PySide6 (GUI) · pytest |

---

## Sumário

1. [Prazos](#1-prazos)
2. [Arquitetura](#2-arquitetura)
3. [Contrato entre os módulos](#3-contrato-entre-os-módulos-leia-antes-de-codar)
4. [Como rodar](#4-como-rodar)
5. [Regras do repositório](#5-regras-do-repositório-eliminatório)
6. [Responsabilidades](#6-responsabilidades)
   - [6.1 Igor: pré-processamento e danos](#61-igor-zanette-pré-processamento-detecção-e-preenchimento-de-danos)
   - [6.2 Marco: Deep Learning](#62-marco-antonio-santolin-deep-learning)
   - [6.3 Marcos Paulo: interface gráfica](#63-marcos-paulo-de-medeiros-interface-gráfica-gui)
   - [6.4 Itens da proposta enviada ao professor](#64-itens-da-proposta-enviada-ao-professor-a-distribuir)
7. [O "algo a mais"](#7-o-algo-a-mais)
8. [Cronograma](#8-cronograma)
9. [Checklist de entrega](#9-checklist-de-entrega-requisitos-do-enunciado)

---

## 1. Prazos

| Data | Marco |
|---|---|
| 16/11 (seg) | Sprint em aula: base do pipeline pronta, DL e GUI iniciados |
| 23/11 (seg) | Sprint em aula: **tudo integrado na GUI**, extras funcionando |
| 24–29/11 | Polimento, testes com fotos variadas, vídeo demo, README, ensaio |
| **30/11 (seg)** | **Apresentação final** (até 10 min; o professor vai testar com fotos dele) |

## 2. Arquitetura

```
           ┌──────────────────────── retrato_vivo.pipeline.restore() ─────────────────────────┐
 arquivo → │ diagnóstico → cor/contraste → detectar danos → apagar danos → acabamento → IA    │ → GUI
           │ [Igor]        [Igor]          [Igor]           [Igor]         [Igor]       [Marco]│  [Marcos]
           └──────────────────────────────────────────────────────────────────────────────────┘
              ↑ cada etapa aplicada fica em result.steps; o motivo de cada decisão, em result.notes
```

```
retrato-vivo/
├── retrato_vivo/
│   ├── io_utils.py      [Igor]  abrir e salvar qualquer foto
│   ├── preprocess.py    [Igor]  diagnóstico, cor, contraste, ruído, nitidez
│   ├── faces.py         [Igor]  detector de rostos (YuNet)
│   ├── damage.py        [Igor]  detecção de riscos, vincos e manchas
│   ├── inpaint.py       [Igor]  preenchimento dos danos
│   ├── visualize.py     [Igor]  mapa de danos, antes/depois, painel de etapas
│   ├── aging.py         [Igor]  efeitos de foto antiga (avaliação + Máquina do tempo)
│   ├── metrics.py       [Igor]  PSNR, SSIM, precisão/revocação
│   ├── pipeline.py      [todos] ordem das etapas: o PONTO DE INTEGRAÇÃO
│   ├── models.py        [todos] registro e download dos modelos pré-treinados
│   ├── cli.py                   linha de comando (testar sem GUI)
│   ├── ai/              [Marco] colorização, super-resolução, rosto...
│   └── gui/             [Marcos Paulo] interface gráfica
├── models/              pesos baixados (NÃO vão para o git)
├── tests/               pytest
└── docs/                este documento e COMO_FUNCIONA.md
```

A explicação detalhada da restauração clássica está em
**[COMO_FUNCIONA.md](COMO_FUNCIONA.md)**.

## 3. Contrato entre os módulos (leia antes de codar)

Quem seguir estas regras consegue plugar seu trabalho sem mexer no código dos outros.

1. **Formato de imagem único:** `np.ndarray`, **BGR**, `uint8`, formato `(H, W, 3)`.
   Sempre carregue com `io_utils.load_image` e salve com `io_utils.save_image`.
   Nunca use `cv2.imread`: ele quebra com acentos no Windows e ignora a rotação de
   fotos de celular.
2. **Etapa nova = função**, sem estado global escondido:
   - *Post stage* (Marco): `def etapa(img: np.ndarray, result: RestorationResult) -> np.ndarray`.
     Pode consultar `result.diagnosis.monochrome`, `result.damage.faces` etc.
   - *Inpainter* alternativo (Marco, opcional): `def inpainter(img, mask_uint8) -> np.ndarray`.
3. **A GUI só fala com `pipeline.restore()`:**

   ```python
   from retrato_vivo import load_image, restore, RestorationConfig
   from retrato_vivo.pipeline import STEP_INFO
   from retrato_vivo.ai import colorize            # exemplo (Marco)

   cfg = RestorationConfig()
   cfg.damage.sensitivity = 0.6                    # slider da GUI
   res = restore(load_image(caminho), cfg,
                 mask_add=pincel, mask_remove=borracha,          # editor de máscara
                 post_stages=[("colorizacao", colorize)])        # etapas de IA ligadas

   res.image          # resultado final
   res.original       # entrada redimensionada (é sobre ela que o usuário pinta a máscara)
   res.steps          # dict ordenado nome -> imagem de cada etapa aplicada
   res.notes          # dict nome -> frase explicando o que a etapa fez ou por que foi pulada
   res.diagnosis      # o que foi medido: monochrome, noise_sigma, cast_strength...
   res.damage.stats   # nº de riscos, vincos, manchas, % da área, rostos protegidos
   res.damage.report  # quantas candidatas cada teste descartou
   res.timings_ms     # tempo de cada fase (mostrar na GUI = critério "desempenho")
   STEP_INFO[nome]    # (título, descrição) de cada etapa, para exibir ao usuário
   ```
   A lista de controles que a GUI pode expor está em
   [COMO_FUNCIONA.md, seção 5](COMO_FUNCIONA.md#5-controles-que-a-interface-pode-oferecer).
   Visualizações prontas: `visualize.damage_map`, `side_by_side`, `steps_panel`.
4. **Modelos pré-treinados:** registre em `retrato_vivo/models.py` (nome, arquivo, URL) e
   use `models.model_path("nome")`. Os pesos **não** vão para o git; cada um baixa com
   `python -m retrato_vivo modelos`.
5. **Toda função nova ganha pelo menos um teste** em `tests/`.

## 4. Como rodar

```bash
git clone https://github.com/MrcSanto/retrato-vivo && cd retrato-vivo
python -m venv .venv && .venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m retrato_vivo modelos                          # baixa os pesos para models/

python -m pytest                                        # testes (devem passar sempre)
python -m retrato_vivo restaurar foto.jpg -o saida --painel --explicar
python -m retrato_vivo restaurar pasta/ -o saida --sensibilidade 0.7
python -m retrato_vivo avaliar foto_boa.jpg -o avaliacao         # métricas objetivas
python -m retrato_vivo envelhecer foto_boa.jpg -o antiga.png     # estraga uma foto para testar
```

## 5. Regras do repositório (eliminatório)

O professor avalia o **envolvimento de cada um pelos commits**.

- **Cada um commita o próprio trabalho, com a própria conta do GitHub.** Ninguém
  sobe o código de outra pessoa.
- Uma branch por tarefa: `igor/vincos`, `marco/colorizacao`, `marcos/gui-mascara`.
  Integre na `main` por Pull Request, e alguém da equipe dá uma olhada antes do merge.
- Commits pequenos e frequentes (várias vezes por semana), com mensagens em português
  e no imperativo: `Adiciona detecção de vincos`.
- Antes de abrir o PR: `python -m pytest` precisa passar.
- **Nunca** commitar: pesos de modelos (`models/`), fotos pessoais, `saida/`, `.venv/`.
- O repositório precisa estar **compartilhado com `rafaelrieder`**
  (Settings → Collaborators). **Responsável: Marco** (dono do repo).

## 6. Responsabilidades

### 6.1 Igor Zanette: pré-processamento, detecção e preenchimento de danos

**Status: ✅ implementado, revisado com 13 fotos reais, testado e na `main` (15/11).
Já está integrado à colorização (Marco) e à interface (Marcos Paulo). Diagnóstico de sépia
de alto contraste corrigido (20/11). Plano de IA do [diagnóstico](DIAGNOSTICO_RESTAURACAO.md)
aplicado (20/11): detector de riscos com IA, LaMa, GPU/CPU.**

Explicação completa, técnicas, métricas e limitações: **[COMO_FUNCIONA.md](COMO_FUNCIONA.md)**.

Resumo do que existe:

- Abertura robusta de arquivos (acentos, rotação de celular, transparência, 16 bits, AVIF via Pillow).
- Diagnóstico da foto e correções **adaptativas**: cada etapa só age se a foto
  precisar, e explica a decisão.
- Detecção de **riscos**, **vincos** e **manchas**, com proteção de rostos (YuNet)
  e relatório de quantas candidatas cada teste descartou.
- Preenchimento por inpainting (Telea ou Navier-Stokes).
- Avaliação objetiva: envelhecer uma foto boa, restaurar e medir (PSNR, SSIM, F1).
- Efeitos de época prontos para a Máquina do tempo (`aging.py`).
- 38 testes automáticos em `tests/test_restauracao.py`.

**Feito:**
- [x] 13 commits desta parte na `main` (15/11).
- [x] Integração com a colorização: a etapa do Marco usa `result.diagnosis.monochrome`.
- [x] Integração com a interface: o editor de máscara do Marcos Paulo usa `mask_add` /
  `mask_remove`, e a interface mostra `steps`, `notes`, `damage.stats` e `timings_ms`.
  O apoio previsto ao editor não foi necessário.
- [x] **Sépia de alto contraste diagnosticada como colorida** (apontado pelo Marco, 20/11):
  além da variação de cor, `diagnose` agora mede quanto da cor fica fora do tom dominante
  (`off_tone_pct`). Fotos reais certas: 12/13 → 13/13 (`foto_rasgada2` era o erro); com 63
  variações sintéticas, 46/76 → 71/76. Detalhes em [COMO_FUNCIONA.md](COMO_FUNCIONA.md).
  Falta conferir a colorização de ponta a ponta com OpenCV 4.x e os modelos baixados.

- [x] **IA na restauração (20/11):** detector de riscos da Microsoft convertido para ONNX
  (`damage_ai.py`), LaMa nas áreas grandes (`inpaint.py`, método `ia`), execução na GPU com
  queda automática para a CPU (`runtime.py`). Também, a pedido do Igor, nas áreas dos colegas:
  `ai/face_restore.py` (GFPGAN) e `ai/ddcolor.py` (Marco), controles novos e navegação de
  pasta na interface (Marcos Paulo). Detalhes e números em
  [DIAGNOSTICO_RESTAURACAO.md](DIAGNOSTICO_RESTAURACAO.md#0-o-que-já-foi-aplicado-2011-pelo-igor-na-rtx-4080).

**Pendências do Igor:**
- [ ] Publicar `scratch_detector.onnx` num Release do repositório e pôr a URL em `models.py`.
- [ ] Rodar os testes com OpenCV 4.x, a versão fixada agora no `requirements.txt`.
  A parte do Igor foi desenvolvida e testada no OpenCV 5.
- [ ] Vinco que cruza cabelo ou outra textura ainda fica com um trecho sem reparo.
- [ ] Detecção de rasgos (`detect_large_losses`) é básica e fica desligada por padrão.

**Ainda sem responsável, com a área do Igor como candidata:** correção de perspectiva
(seção 6.4), Retrato Vivo e Máquina do tempo (seção 7).

---

### 6.2 Marco Antonio Santolin: Deep Learning

**Status: 🔄 colorização pronta e testada (16/11); super-resolução a seguir.**
Pasta: `retrato_vivo/ai/`. É aqui que está o requisito
"Deep Learning" do enunciado, então é a parte mais visível da nota de complexidade.

**Passo a passo:**

1. Ler a [seção 3](#3-contrato-entre-os-módulos-leia-antes-de-codar) e rodar
   `python -m retrato_vivo restaurar foto.jpg --painel --explicar` para ver o pipeline funcionando.
2. ✅ **Colorização (prioridade 1).** Feito em `ai/colorize.py`: `colorize` (etapa pronta),
   `make_colorize_stage(intensidade)` (para o slider da GUI) e `colorize_image(img, intensidade)`.
   Linha de comando: `restaurar foto.jpg --colorir --intensidade-cor 0.8`. 10 testes em `tests/test_ai.py`.
   ✅ **Corrigido (Igor, 20/11):** foto sépia de alto contraste era diagnosticada como colorida
   (`chroma_std` passava de 6) e a colorização era pulada. Agora `diagnose` também aceita
   como sépia a foto cuja cor está quase toda num tom só (`off_tone_pct < 5`). Plano original,
   para referência:
   Modelo de Zhang et al. (ECCV 2016) via `cv2.dnn`:
   - Arquivos: `colorization_deploy_v2.prototxt`, `colorization_release_v2.caffemodel`,
     `pts_in_hull.npy` (repositório `richzhang/colorization`). Registrar em `models.py`.
   - Criar `retrato_vivo/ai/colorize.py` com `colorize(img, result) -> img`.
     Colorir **só se** `result.diagnosis.monochrome` for verdadeiro (foto colorida passa direto).
     Atenção: foto sépia chega em **cinza neutro**; quem devolve a cor é esta etapa.
   - Pipeline: BGR → LAB, canal L redimensionado para 224×224, menos 50 → rede → canais ab
     → redimensionar para o tamanho original → juntar com o L original → BGR.
   - Parâmetro de intensidade (0 a 1) para misturar com a versão cinza: a cor fica menos "berrante".
3. **Super-resolução (prioridade 2).** ESPCN/FSRCNN (`.pb`, ×2/×4) via `cv2.dnn.readNet`
   (o módulo `dnn_superres` está no *contrib*, que não instalamos). Etapa
   `upscale(img, result)`. Medir o tempo, porque ×4 em foto grande é lento.
4. **Restauração de rosto (prioridade 3, opcional).** Recortar os rostos usando
   `result.damage.faces` (o YuNet já roda no pipeline), aplicar um modelo de rosto em
   ONNX (ex.: GFPGAN ou CodeFormer) e colar de volta com `cv2.seamlessClone`.
   Testar antes se o OpenCV carrega o ONNX escolhido.
5. **Inpainting com DL (opcional).** As fotos rasgadas (pedaços faltando) estão fora do
   alcance do preenchimento clássico. Um modelo como o LaMa pode ser plugado em
   `restore(..., inpainter=...)`.
6. **CUDA (opcional).** O `opencv-python` do pip **não tem CUDA**. Só vale a pena se der
   tempo de compilar o OpenCV com CUDA. O enunciado aceita "DL **e/ou** CUDA", então DL basta.
7. Testes em `tests/test_ai.py`, usando `pytest.mark.skipif` quando o modelo não foi baixado.
8. **Responsável por compartilhar o repo com `rafaelrieder`.**

---

### 6.3 Marcos Paulo de Medeiros: interface gráfica (GUI)

**Status: ✅ itens 1 a 8 prontos e testados (18/11), já com o polimento: restauração
refeita ao mexer nos controles, e desfazer e zoom no editor de máscara. Falta só o item 9
(abas dos extras), que depende dos extras.** Como usar: seção "Interface gráfica" do
[README](../README.md). Pasta: `retrato_vivo/gui/`. Vale 1,0 ponto ("qualidade da
interface e desempenho"), e é por ela que o professor vai testar com fotos dele.

**Framework:** **PySide6** (Qt): visual profissional, threads e arrastar-e-soltar prontos.
No `requirements.txt` entrou o `PySide6-Essentials`, que traz só QtCore, QtGui e QtWidgets.
Ponto de entrada: `python -m retrato_vivo.gui [foto]`. Testes em `tests/test_gui.py`, que
rodam sem tela.

**Funcionalidades, em ordem de prioridade:**

1. ✅ Abrir imagem (botão + **arrastar e soltar**), com todos os formatos de
   `io_utils.SUPPORTED_EXTENSIONS`. Carregar **sempre** via `load_image`.
2. ✅ Botão "Restaurar", chamando `restore()` numa **`QThread`** (a interface não pode travar) e
   mostrando uma barra de progresso. Mexer nos controles também refaz a restauração, 0,4 s
   depois da última mexida, sempre uma de cada vez.
3. ✅ **Comparação antes/depois com slider** arrastável sobre a imagem, com os rótulos
   "Antes" e "Depois".
4. ✅ Lista ou abas com **cada etapa** (`res.steps`), com o título e a descrição de
   `STEP_INFO` e a frase de `res.notes` ("ruído medido 0,7: foto limpa, etapa pulada").
   É isso que mostra o "processamento de imagens claro" que o enunciado pede.
5. ✅ Painel de controles: ver a
   [tabela de controles](COMO_FUNCIONA.md#5-controles-que-a-interface-pode-oferecer).
   Inclui a colorização (caixa "Colorir foto P&B" e intensidade). Etapa de IA nova entra em
   `ControlsPanel.post_stages()`, em `gui/controls.py`.
6. ✅ **Editor de máscara:** sobre `res.original`, pincel (adicionar dano) e borracha (remover
   falso positivo), com tamanho de pincel ajustável. Gera `mask_add` e `mask_remove`
   (uint8, mesmo tamanho de `res.original`) e chama `restore()` de novo. Mostrar
   `steps["mapa_de_danos"]` como camada. É essencial: resolve os casos que a detecção
   automática não pega. O que a borracha desmarca continua colorido no mapa de danos,
   porque o mapa pinta as detecções por tipo; o reparo respeita a borracha. Tem desfazer
   (Ctrl+Z, os 10 últimos traços) e zoom de até 8 vezes (roda do mouse; o botão direito
   arrasta a foto ampliada).
7. ✅ Salvar resultado, mais "exportar comparativo" (`visualize.side_by_side`).
8. ✅ Barra de status com `res.damage.stats` e `res.timings_ms`.
9. Abas dos extras (ver seção 7).

**Também é responsabilidade do Marcos Paulo:** README final (prints e link do vídeo) e
gravação do vídeo demo de 30 a 60 s.

---

### 6.4 Itens da proposta enviada ao professor (a distribuir)

A proposta enviada ao professor prometeu dois itens que ainda não tinham tarefa, e
definiu a super-resolução com outros modelos. **A equipe precisa definir os responsáveis.**

| Item | O que fazer | Técnicas | Área sugerida | Responsável |
|---|---|---|---|---|
| **Correção de perspectiva** ("foto de uma foto") | Quem fotografa uma foto impressa com o celular tem o papel torto, em trapézio, e com fundo em volta. A etapa acha as 4 bordas do papel, "desentorta" e recorta. | Canny/limiarização, `findContours`, `approxPolyDP` (maior quadrilátero), ordenar cantos, `getPerspectiveTransform` + `warpPerspective` | Igor (pré-processamento) ou Marcos Paulo (modo da GUI) | **A definir** |
| **Comparação de bordas Canny vs. HED** | Mostrar lado a lado as bordas da foto antes e depois da restauração, pelo método clássico (Canny) e por Deep Learning (HED). Mostra o ganho da restauração e a diferença entre PI clássico e DL. | `cv2.Canny`; HED (Xie & Tu, ICCV 2015, Caffe) via `cv2.dnn`, com a camada `Crop` customizada | Marco (Deep Learning) | **A definir** (sugestão: Marco) |
| **Super-resolução: modelos** | A proposta citava ESPCN/EDSR via `dnn_superres`, mas esse módulo está no *contrib*, que não instalamos. Usar os mesmos `.pb` via `cv2.dnn`: **ESPCN** como padrão (rápido) e **EDSR** como "qualidade máxima" (lento). | `cv2.dnn.readNetFromTensorflow`, ×2/×4 | Marco | Marco (já na 6.2) |

**Correção de perspectiva, detalhes para quem pegar:**
- Contrato: `def corrigir_perspectiva(img) -> tuple[np.ndarray, bool]`, que devolve a foto
  corrigida e se achou o papel. Se não achar um quadrilátero confiável (área mínima, ângulos
  razoáveis), devolve a foto intacta: uma foto digitalizada normal não pode ser estragada.
- Entra **antes** do diagnóstico em `pipeline.restore()`, ligada por uma opção em
  `RestorationConfig` (desligada por padrão) e anotada em `result.steps` / `result.notes`.
- GUI: caixa "É foto de uma foto impressa", mostrando os 4 cantos detectados, que o usuário
  pode arrastar se errar.
- Testes: desenhar um retângulo claro em perspectiva sobre um fundo escuro e conferir que a
  saída fica retangular; foto sem papel volta intacta.

**Canny vs. HED, detalhes:**
- Modelo: `deploy.prototxt` + `hed_pretrained_bsds.caffemodel` (56 MB), registrar em `models.py`.
  Lê via `readNetFromCaffe`, mais um motivo para o OpenCV 4.x.
- Função em `ai/edges.py`: `bordas_canny(img)`, `bordas_hed(img)` e `painel_bordas(antes, depois)`
  (2×2: Canny/HED × antes/depois). Não altera a foto: é uma visualização, como `visualize.py`.
- GUI e CLI: aba "Bordas" / opção `--bordas`.

---

## 7. O "algo a mais"

O professor pediu que o sistema faça algo além de restaurar. **Decisão (15/11):**

| | Ideia | O que faz | Técnicas | Responsável |
|---|---|---|---|---|
| ⭐ **Destaque** | **Retrato Vivo (foto em movimento)** | Transforma a foto restaurada num vídeo curto com **efeito 3D de paralaxe**: a câmera "passeia" e a pessoa se destaca do fundo. Dá nome ao projeto. | Profundidade com DL (Depth Anything V2 Small, no lugar do MiDaS), camadas por Otsu, fundo escondido com LaMa, `cv2.remap`, MP4/GIF | ✅ **Igor (20/11)** |
| **Bônus** | **Máquina do tempo** | O inverso da restauração: a selfie do professor vira foto de 1900, 1920, 1950, 1970 ou Polaroid dos anos 90. | Efeitos de `aging.py` em presets por época, molduras, data; “ida e volta” com a restauração | ✅ **Igor (20/11)** |

Os dois foram implementados pelo Igor em 20/11, com abas próprias na interface. Explicação e
roteiro para a apresentação: **[DIFERENCIAL.md](DIFERENCIAL.md)**.

### O que já existe da parte do Igor para cada extra

**Máquina do tempo.** Os efeitos estão prontos e testados em `retrato_vivo/aging.py`.
Uma época é só uma sequência deles (exemplo no topo do arquivo):

| Grupo | Efeitos |
|---|---|
| Tom e cor | `black_and_white`, `sepia`, `fade`, `contrast`, `saturation`, `color_cast` |
| Lente e filme | `soften`, `vignette`, `grain` |
| Danos | `add_scratches`, `add_spots` |
| Acabamento | `crop_square`, `add_border`, `polaroid_frame`, `date_stamp` |

Falta para quem pegar: os presets de cada época (1900, 1920, 1950, 1970, Polaroid 1990),
um comando ou função `viajar(foto, epoca)` e a aba na GUI. Bônus para a demo: envelhecer
a selfie do professor e depois restaurá-la com o pipeline (o "ida e volta" já é testado
em `test_aged_photo_is_restorable`).

**Retrato Vivo.** Usa da parte do Igor:

- `restore(foto).image`: a foto restaurada é a entrada do efeito 3D.
- `result.damage.faces`: caixas dos rostos (YuNet), úteis para manter a pessoa em
  primeiro plano e centralizar o movimento da câmera.
- `inpaint(img, mascara, "ns")`: preenche as faixas de fundo reveladas quando a câmera
  se move. Testei Telea, Navier-Stokes e uma versão multiescala em buracos desse
  formato: os três ficaram com erro parecido, e nenhum recria detalhe (o fundo vira um
  borrão). Por isso a versão multiescala não entrou. Para disfarçar, mantenha o
  movimento pequeno (as faixas ficam estreitas) e preencha a máscara *antes* de mover.

Falta para quem pegar: estimativa de profundidade (MiDaS via `cv2.dnn`, registrar em
`models.py`), separar camadas pela profundidade, mover a câmera, gerar os quadros e
exportar MP4/GIF.

Ideias que ficaram de fora (só se sobrar tempo): estimar a idade das pessoas e dizer
"hoje teria ~86 anos"; agrupar a mesma pessoa em várias fotos (`cv2.FaceRecognizerSF`);
colorização guiada por clique; laudo de restauração em HTML/PDF.

## 8. Cronograma

| Período | Igor | Marco | Marcos Paulo |
|---|---|---|---|
| 15–16/11 | ✅ Restauração clássica, testes, documentação | Colorização + registro de modelos | Esqueleto da GUI: abrir, restaurar, antes/depois |
| 17–22/11 | Ajustes com mais fotos; apoio ao editor de máscara | Super-resolução; rosto ou inpainting DL (opcional) | Etapas, controles, editor de máscara, QThread |
| 17–22/11 | **Extras (responsáveis a definir): Retrato Vivo e Máquina do tempo** | | |
| 17–22/11 | **Itens da proposta (seção 6.4, responsáveis a definir): correção de perspectiva e Canny vs. HED** | | |
| **23/11** | **Integração na GUI (todos juntos em aula)** | | |
| 24–27/11 | Ajustes finos | Ajustes finos | README |
| 28–29/11 | Ensaio da apresentação + **vídeo demo** + teste com fotos nunca vistas | | |
| **30/11** | **Apresentação** | | |

## 9. Checklist de entrega (requisitos do enunciado)

- [ ] **OpenCV + Deep Learning:** YuNet ✅, colorização, super-resolução (ESPCN/EDSR), bordas HED, profundidade (MiDaS)
- [ ] **Tudo o que a proposta enviada ao professor prometeu:** correção de perspectiva, Canny vs. HED,
      slider antes/depois (ver seção 6.4)
- [x] **GUI** funcional (PySide6)
- [ ] **Aplicabilidade prática:** restaurar fotos de família, de forma acessível para leigos
- [x] **Técnicas de PI claras:** morfologia, segmentação, histerese, inpainting, NLM, CLAHE
- [ ] **Vídeo demo de 30 a 60 s**, com link no README
- [x] **Flexibilidade de arquivos:** qualquer formato, tamanho, P&B/colorida, 16 bits, alfa, acentos, celular
- [ ] Repo **compartilhado com `rafaelrieder`**
- [ ] **Commits dos três, equilibrados**
- [ ] Repo só com código-fonte e dependências (sem pesos nem fotos pessoais)
- [ ] Ensaio de **até 10 min**, com fotos de teste que nunca foram usadas no desenvolvimento
