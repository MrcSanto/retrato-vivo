# Retrato Vivo: fluxo de trabalho da equipe

> Documento vivo. **Leia inteiro antes de começar a sua parte** e atualize a
> seção "Status" da sua área quando terminar algo.

| | |
|---|---|
| **Disciplina** | CCC309 – Processamento de Imagens e Visão Computacional (UPF, 2026/2) |
| **Professor** | Dr. Rafael Rieder (GitHub: `rafaelrieder`) |
| **Tema** | Restaurador de fotos antigas, com dois extras (ver [seção 7](#7-o-algo-a-mais)) |
| **Equipe** | Igor Zanette · Marco Antonio Santolin · Marcos Paulo de Medeiros |
| **Stack** | Python 3.12 · OpenCV 5 (`cv2`, `cv2.dnn`) · NumPy · PySide6 (GUI) · pytest |

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
│   ├── aging.py         [Igor]  envelhecimento sintético (para avaliar)
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

**Status: ✅ implementado, revisado com 13 fotos reais e testado (15/11).**

Explicação completa, técnicas, métricas e limitações: **[COMO_FUNCIONA.md](COMO_FUNCIONA.md)**.

Resumo do que existe:

- Abertura robusta de arquivos (acentos, rotação de celular, transparência, 16 bits, AVIF via Pillow).
- Diagnóstico da foto e correções **adaptativas**: cada etapa só age se a foto
  precisar, e explica a decisão.
- Detecção de **riscos**, **vincos** e **manchas**, com proteção de rostos (YuNet)
  e relatório de quantas candidatas cada teste descartou.
- Preenchimento por inpainting (Telea ou Navier-Stokes).
- Avaliação objetiva: envelhecer uma foto boa, restaurar e medir (PSNR, SSIM, F1).
- 29 testes automáticos.

**Pendências do Igor:**
- [ ] Fazer os commits desta parte (em vários commits, por arquivo/assunto).
- [ ] Vinco que cruza cabelo ou outra textura ainda fica com um trecho sem reparo.
- [ ] Detecção de rasgos (`detect_large_losses`) é básica e fica desligada por padrão.
- [ ] Apoiar o Marcos Paulo no editor de máscara (`mask_add` / `mask_remove`).

---

### 6.2 Marco Antonio Santolin: Deep Learning

**Status: ⏳ não iniciado.** Pasta: `retrato_vivo/ai/`. É aqui que está o requisito
"Deep Learning" do enunciado, então é a parte mais visível da nota de complexidade.

**Passo a passo:**

1. Ler a [seção 3](#3-contrato-entre-os-módulos-leia-antes-de-codar) e rodar
   `python -m retrato_vivo restaurar foto.jpg --painel --explicar` para ver o pipeline funcionando.
2. **Colorização (prioridade 1).** Modelo de Zhang et al. (ECCV 2016) via `cv2.dnn`:
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
   Testar antes se o OpenCV 5 carrega o ONNX escolhido.
5. **Inpainting com DL (opcional).** As fotos rasgadas (pedaços faltando) estão fora do
   alcance do preenchimento clássico. Um modelo como o LaMa pode ser plugado em
   `restore(..., inpainter=...)`.
6. **CUDA (opcional).** O `opencv-python` do pip **não tem CUDA**. Só vale a pena se der
   tempo de compilar o OpenCV com CUDA. O enunciado aceita "DL **e/ou** CUDA", então DL basta.
   O OpenCV 5 também avisa `Targets are not supported by the new graph engine` (pode ignorar).
7. Testes em `tests/test_ai.py`, usando `pytest.mark.skipif` quando o modelo não foi baixado.
8. **Responsável por compartilhar o repo com `rafaelrieder`.**

---

### 6.3 Marcos Paulo de Medeiros: interface gráfica (GUI)

**Status: ⏳ não iniciado.** Pasta: `retrato_vivo/gui/`. Vale 1,0 ponto ("qualidade da
interface e desempenho"), e é por ela que o professor vai testar com fotos dele.

**Sugestão de framework:** **PySide6** (Qt): visual profissional, threads e
arrastar-e-soltar prontos. Adicionar `PySide6` ao `requirements.txt`. Ponto de entrada:
`python -m retrato_vivo.gui`.

**Funcionalidades, em ordem de prioridade:**

1. Abrir imagem (botão + **arrastar e soltar**), com todos os formatos de
   `io_utils.SUPPORTED_EXTENSIONS`. Carregar **sempre** via `load_image`.
2. Botão "Restaurar", chamando `restore()` numa **`QThread`** (a interface não pode travar) e
   mostrando uma barra de progresso.
3. **Comparação antes/depois com slider** arrastável sobre a imagem.
4. Lista ou abas com **cada etapa** (`res.steps`), com o título e a descrição de
   `STEP_INFO` e a frase de `res.notes` ("ruído medido 0,7: foto limpa, etapa pulada").
   É isso que mostra o "processamento de imagens claro" que o enunciado pede.
5. Painel de controles: ver a
   [tabela de controles](COMO_FUNCIONA.md#5-controles-que-a-interface-pode-oferecer).
6. **Editor de máscara:** sobre `res.original`, pincel (adicionar dano) e borracha (remover
   falso positivo), com tamanho de pincel ajustável. Gera `mask_add` e `mask_remove`
   (uint8, mesmo tamanho de `res.original`) e chama `restore()` de novo. Mostrar
   `steps["mapa_de_danos"]` como camada. É essencial: resolve os casos que a detecção
   automática não pega.
7. Salvar resultado, mais "exportar comparativo" (`visualize.side_by_side`).
8. Barra de status com `res.damage.stats` e `res.timings_ms`.
9. Abas dos extras (ver seção 7).

**Também é responsabilidade do Marcos Paulo:** README final (prints e link do vídeo) e
gravação do vídeo demo de 30 a 60 s.

---

## 7. O "algo a mais"

O professor pediu que o sistema faça algo além de restaurar. **Decisão (15/11):**

| | Ideia | O que faz | Técnicas | Responsável |
|---|---|---|---|---|
| ⭐ **Destaque** | **Retrato Vivo (foto em movimento)** | Transforma a foto restaurada num vídeo curto com **efeito 3D de paralaxe**: a câmera "passeia" e a pessoa se destaca do fundo. Dá nome ao projeto. | Estimativa de profundidade com DL (MiDaS small, ONNX via `cv2.dnn`), *warping* por camadas de profundidade, inpainting das bordas reveladas, exportação em MP4/GIF (`cv2.VideoWriter`) | **A definir** |
| **Bônus** | **Máquina do tempo** | O inverso da restauração: a selfie do professor vira foto de 1900, 1920, 1950, 1970 ou Polaroid dos anos 90. | Combinar os efeitos de `aging.py` (sépia, grão, vinheta, riscos) em presets por época, mais molduras e data | **A definir** |

A equipe precisa definir os responsáveis na aula de 16/11, de forma a manter os
commits equilibrados.

Ideias que ficaram de fora (só se sobrar tempo): estimar a idade das pessoas e dizer
"hoje teria ~86 anos"; agrupar a mesma pessoa em várias fotos (`cv2.FaceRecognizerSF`);
colorização guiada por clique; laudo de restauração em HTML/PDF.

## 8. Cronograma

| Período | Igor | Marco | Marcos Paulo |
|---|---|---|---|
| 15–16/11 | ✅ Restauração clássica, testes, documentação | Colorização + registro de modelos | Esqueleto da GUI: abrir, restaurar, antes/depois |
| 17–22/11 | Ajustes com mais fotos; apoio ao editor de máscara | Super-resolução; rosto ou inpainting DL (opcional) | Etapas, controles, editor de máscara, QThread |
| 17–22/11 | **Extras (responsáveis a definir): Retrato Vivo e Máquina do tempo** | | |
| **23/11** | **Integração na GUI (todos juntos em aula)** | | |
| 24–27/11 | Ajustes finos | Ajustes finos | README |
| 28–29/11 | Ensaio da apresentação + **vídeo demo** + teste com fotos nunca vistas | | |
| **30/11** | **Apresentação** | | |

## 9. Checklist de entrega (requisitos do enunciado)

- [ ] **OpenCV + Deep Learning:** YuNet ✅, colorização, super-resolução, profundidade (MiDaS)
- [ ] **GUI** funcional (PySide6)
- [ ] **Aplicabilidade prática:** restaurar fotos de família, de forma acessível para leigos
- [x] **Técnicas de PI claras:** morfologia, segmentação, histerese, inpainting, NLM, CLAHE
- [ ] **Vídeo demo de 30 a 60 s**, com link no README
- [x] **Flexibilidade de arquivos:** qualquer formato, tamanho, P&B/colorida, 16 bits, alfa, acentos, celular
- [ ] Repo **compartilhado com `rafaelrieder`**
- [ ] **Commits dos três, equilibrados**
- [ ] Repo só com código-fonte e dependências (sem pesos nem fotos pessoais)
- [ ] Ensaio de **até 10 min**, com fotos de teste que nunca foram usadas no desenvolvimento
