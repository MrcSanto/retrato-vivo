# Diagnóstico: como deixar a restauração melhor

Feito em 20/11 com as 13 fotos de `imagens/`, depois de a colorização e a interface ficarem
prontas. Objetivo: entender por que o resultado ainda é fraco (principalmente em fotos
rasgadas) e escolher o que fazer até a apresentação de 30/11.

## 0. O que já foi aplicado (20/11, pelo Igor, na RTX 4080)

| Item | Situação | Resultado medido |
|---|---|---|
| P1. LaMa no preenchimento | ✅ `inpaint.py` (método `ia`, padrão) | Áreas pintadas com o pincel são reconstruídas em vez de borradas; ~60 ms na GPU por área |
| P2. Detector de riscos com IA | ✅ `damage_ai.py` + `scripts/exportar_detector_riscos.py` | Na `foto_risco3`, 57 riscos a mais e a foto sai limpa; 0% de falso alarme na `foto_boa1` |
| P3. Restauração de rosto | ✅ `ai/face_restore.py` (GFPGAN, opcional) | Rostos nítidos e naturais; ~60 a 120 ms na GPU |
| P4. DDColor | ✅ `ai/ddcolor.py` (padrão da interface; Zhang continua como opção) | Melhor que Zhang em 4 de 5 fotos antigas; erra no daguerreótipo com moldura (rosa/azul) |
| P4. Real-ESRGAN | ⏸️ não feito: a super-resolução é do Marco e está na branch `marco/super-resolucao` | |
| P5. GPU (CUDA) | ✅ `runtime.py` (ONNX Runtime): GPU quando houver, CPU nos outros PCs | Cadeia completa: ~0,4 s na GPU, ~4 s só na CPU |
| Controles mortos | ✅ "Reduzir ruído" acima de 1 agora age; "Rasgos" e "Proteger rostos" escondidos; novos: "Riscos com IA", método "IA (LaMa)", "Restaurar rostos", modelo de cor | |
| Detectar rasgos sozinho | ❌ descartado: uma regra "claro e liso" marcaria 31% de um fundo de estúdio. Rasgo = pincel + LaMa | |

A decisão da seção 5 foi tomada pela opção **A** (OpenCV 4.x + ONNX Runtime). O `AGENTS.md`
ainda diz "carregue com `cv2.dnn`" e precisa ser atualizado pela equipe.

## 1. Resumo

| Problema | Causa | Solução recomendada |
|---|---|---|
| **Muitos riscos e vincos não são detectados** | A detecção é feita com regras e limiares ajustados à mão (morfologia). Riscos longos se quebram em pedaços e caem no teste "curto demais"; riscos sobre bordas ou textura são descartados de propósito | **Detector de riscos com IA** (U-Net do *Bringing Old Photos Back to Life*, Microsoft, CVPR 2020), somado ao detector atual |
| **Rasgos praticamente não melhoram** | (a) as áreas onde o papel aparece não são detectadas; (b) mesmo marcadas, o preenchimento clássico (Telea/Navier-Stokes) só borra áreas grandes | **Preenchimento com IA (LaMa)** + **marcar o rasgo com o pincel** do editor de máscara |
| **Controles da interface que não fazem nada** | Etapas que são puladas na maioria das fotos, ou detectores que quase nunca agem | Tirar ou trocar esses controles (seção 4) |
| Rostos e detalhes continuam moles | Nenhuma etapa reconstrói detalhe | (opcional) Restauração de rosto com IA (GFPGAN/CodeFormer) e super-resolução Real-ESRGAN |

O que funciona bem e deve ficar: o diagnóstico da foto (cor, ruído, contraste), a correção
de cor e contraste, a detecção de vincos e riscos claros isolados, a proteção de rostos e a
colorização.

## 2. O que foi medido

### 2.1 Controles da interface: quanto cada um muda a foto final

% de pixels que mudam ao mexer no controle (a partir das configurações padrão):

| Controle | foto_risco3 | foto_rasgada1 | foto_boa1 | Veredito |
|---|---|---|---|---|
| Sensibilidade 0,5 → 0,8 | 2,6% | 7,0% | 0,8% | age pouco |
| Riscos (desligar) | 0,7% | 0,4% | 0,2% | age pouco |
| Vincos (desligar) | 2,0% | 5,5% | 0,0% | age |
| Manchas (desligar) | 0,7% | 2,0% | 0,4% | age pouco |
| **Rasgos e áreas brancas (ligar)** | **0,0%** | **0,0%** | 0,7% | **não faz nada, nem na foto rasgada** |
| Proteger rostos (desligar) | 0,0% | 0,2% | 0,0% | quase nada |
| Método Telea → Navier-Stokes | 1,0% | 1,9% | 0,3% | diferença invisível |
| Tirar amarelado (desligar) | 31,8% | 98,6% | 0,0% | age |
| Recuperar contraste (desligar) | 79,0% | 85,7% | 42,3% | age |
| **Reduzir ruído (desligar ou força ×2)** | **0,0%** | **0,0%** | **0,0%** | **não faz nada** |
| Contraste local (desligar) | 56,8% | 63,2% | 83,1% | age |
| Nitidez (desligar) | 11,0% | 39,2% | 27,6% | age |

Por que os dois controles mortos não agem:
- **Rasgos e áreas brancas:** o detector (`damage.detect_large_losses`) exige branco quase puro
  (≥ 241) e liso. O papel exposto de uma foto rasgada tem textura e fica entre 200 e 230.
- **Reduzir ruído:** a força é calculada a partir do ruído medido, e o slider só multiplica essa
  dose. Nas fotos do teste o ruído é baixo, a dose é zero e zero vezes qualquer coisa é zero.

### 2.2 Rasgos: preenchimento clássico × LaMa

Na `foto_rasgada1`, com a mesma máscara (as áreas de papel exposto, 13,7% da foto):

| Método | Tempo (CPU) | Resultado |
|---|---|---|
| Telea (atual) | 24 ms | "cata-ventos" borrados, claramente artificiais |
| Navier-Stokes | 25 ms | igual ao Telea |
| **LaMa** | 2,8 s | **textura plausível de fundo**, a emenda quase não aparece |

O LaMa ganha com folga. O problema que sobra é a **máscara**: uma regra simples
("claro e sem cor") marcou partes do rosto por engano, não pegou um dos pedaços e, na
`foto_rasgada2`, não marcou nada. Detectar rasgo automaticamente é difícil. Para a
apresentação, o caminho confiável é **o usuário pintar o rasgo** no editor de máscara.

### 2.3 Compatibilidade com o OpenCV

| Modelo | OpenCV 4.14 (o fixado no projeto) | OpenCV 5.0 |
|---|---|---|
| Colorização Zhang (Caffe), atual | ✅ | ❌ (o OpenCV 5 removeu o leitor de Caffe) |
| LaMa do OpenCV Zoo (`inpainting_lama_2025jan.onnx`) | ❌ não carrega | ✅ |
| LaMa ONNX comunitário (`Carve/LaMa-ONNX`) | ❌ não carrega | ✅ |
| YuNet (rostos) | ✅ | ✅ |

**Conflito:** com o `cv2.dnn` não dá para ter a colorização atual e o LaMa na mesma versão
do OpenCV. Ver a decisão na seção 5.

## 3. Plano por prioridade

### P1. Preenchimento com IA (LaMa): o maior ganho visível
- **O que:** um *inpainter* LaMa plugado em `restore(..., inpainter=...)` (o contrato já existe).
  Usar o LaMa nas áreas grandes (rasgos, pedaços faltando) e manter o Telea nos riscos finos,
  que já fica bom e é instantâneo.
- **Como:** recortar a região ao redor de cada buraco, reduzir para 512×512, rodar e colar de
  volta só dentro da máscara. Na interface, uma caixa "Preencher com IA (LaMa)".
- **Modelo:** big-lama (Suvorov et al., WACV 2022), licença Apache-2.0. Há versão no OpenCV Zoo e
  versões ONNX prontas no Hugging Face.
- **Custo:** cerca de 3 s por foto na CPU. Na GPU (RTX 4080), bem menos.
- **Área:** Igor (`inpaint.py` é da área dele).

### P2. Detector de riscos com IA
- **O que:** a U-Net de detecção de riscos do *Bringing Old Photos Back to Life* (Wan et al.,
  CVPR 2020), treinada em fotos antigas reais anotadas. Ela devolve uma máscara de riscos.
- **Como:** os pesos oficiais são PyTorch (licença MIT). É preciso exportar para ONNX com
  `torch.onnx.export` (o PyTorch já está instalado no computador do Igor). A máscara da rede é
  **somada** à do detector clássico, que continua cuidando de manchas, vincos e da proteção de rostos.
- **Risco:** é o item mais trabalhoso (converter, validar e calibrar). Se a exportação travar,
  ficar só com o P1 e com o pincel.
- **Área:** Igor (`damage.py`), ou Marco se ele terminar a super-resolução antes.

### P3. Restauração de rosto (opcional, alto impacto visual)
- **O que:** GFPGAN ou CodeFormer em ONNX, aplicado só nos rostos que o YuNet já encontra, e
  colado de volta com `cv2.seamlessClone`. Em retratos antigos é o que mais impressiona.
- **Atenção:** são modelos grandes (300 a 500 MB). O CodeFormer tem licença não comercial (serve
  para trabalho acadêmico); conferir antes de distribuir.
- **Área:** Marco (`ai/`, já previsto como "rosto" no plano).

### P4. Trocas de modelo nas etapas de IA que já existem (opcional)
- **Colorização:** DDColor (ICCV 2023, Apache-2.0) é bem mais realista que o modelo de Zhang
  (2016), que tende a manchas avermelhadas, como no fundo da `foto_risco3`. Existe em ONNX.
  Bônus: sem Caffe, o conflito da seção 2.3 deixa de existir.
- **Super-resolução:** Real-ESRGAN x4plus (BSD-3) em ONNX recupera mais textura que ESPCN/EDSR em
  fotos degradadas.
- **Área:** Marco.

### P5. GPU (atende ao "CUDA" do enunciado)
Os modelos acima ficam rápidos na RTX 4080 com o `onnxruntime-gpu`
(`pip install onnxruntime-gpu[cuda,cudnn]`, provedor `CUDAExecutionProvider`). Com isso o
projeto cumpre "Deep Learning **e** CUDA", sem precisar compilar o OpenCV.

## 4. Interface: controles a revisar (área do Marcos Paulo)

| Controle | Sugestão |
|---|---|
| Rasgos e áreas brancas | Trocar por **"Preencher com IA (LaMa)"** (P1) e orientar a pintar o rasgo com o pincel |
| Reduzir ruído (caixa e slider) | Mostrar "foto sem ruído: etapa não necessária" quando a dose for zero, ou esconder o slider. Do lado do Igor: fazer o slider definir uma dose mínima |
| Método Telea / Navier-Stokes | Tirar da interface; a diferença é invisível. Se o LaMa entrar, virar "Clássico (rápido) / IA (melhor)" |
| Proteger rostos | Deixar sempre ligado e tirar da interface |
| Riscos / Vincos / Manchas | Manter, mas agrupar em "Detectar automaticamente" |

Já feito em 20/11: **navegação pelas fotos da pasta** (◀ ▶ e uma lista na barra de
ferramentas, que se atualiza sozinha quando fotos são colocadas em `imagens/`).

## 5. Decisão que a equipe precisa tomar: como rodar os modelos novos

| | A. OpenCV 4.x + `onnxruntime` (recomendada) | B. Migrar para OpenCV 5 |
|---|---|---|
| Colorização atual | continua funcionando | precisa trocar por DDColor ONNX (não testado no OpenCV 5) |
| LaMa, GFPGAN, Real-ESRGAN | rodam no `onnxruntime` | LaMa testado ✅; os outros não testados |
| GPU (CUDA) | `onnxruntime-gpu` pronto | OpenCV 5 do pip não tem CUDA |
| Dependência nova | `onnxruntime` / `onnxruntime-gpu` | nenhuma |
| Regra do `AGENTS.md` | precisa mudar ("carregue com `cv2.dnn`") | mantida |

**Recomendação: A.** Funciona com o que já existe, dá GPU de graça e o OpenCV continua sendo a
base de todo o processamento de imagem (diagnóstico, morfologia, segmentação, inpainting
clássico, YuNet). O `onnxruntime` entra só para rodar as redes pesadas.

## 6. Escopo realista até 30/11 (10 dias)

1. **P1 (LaMa) + pincel para rasgos:** resolve a queixa principal e dá um antes/depois forte na demo.
2. **Revisão dos controles mortos** (seção 4): rápida e melhora a nota de interface.
3. **P2 (detector de riscos com IA)**, se a exportação para ONNX funcionar até 25/11.
4. P3 e P4 só se sobrar tempo.

## Fontes

- LaMa: [advimman/lama](https://github.com/advimman/lama), [OpenCV Zoo: inpainting_lama](https://huggingface.co/opencv/inpainting_lama/raw/main/README.md), [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX), [licença Apache-2.0 (big-lama)](https://huggingface.co/smartywu/big-lama)
- MI-GAN (alternativa leve ao LaMa): [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN)
- Detector de riscos: [Bringing Old Photos Back to Life (artigo)](https://arxiv.org/pdf/2004.09484), [código](https://github.com/davvid/Bringing-Old-Photos-Back-to-Life)
- Rosto: [CodeFormer (artigo)](https://arxiv.org/pdf/2206.11253), [modelos ONNX de restauração de rosto](https://uithub.com/hpc203/Face-restoration-models), [facerestore-onnx](https://huggingface.co/yuvraj108c/facerestore-onnx/tree/main)
- Colorização: [DDColor](https://github.com/piddnad/ddcolor), [DDColor ONNX](https://github.com/instant-high/DDColor-onnx)
- Super-resolução: [Real-ESRGAN ONNX](https://huggingface.co/SceneWorks/real-esrgan-onnx)
- GPU: [onnxruntime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [instalação do ONNX Runtime](https://onnxruntime.ai/docs/install/)
