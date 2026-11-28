# Roteiro da apresentação (30/11)

**No máximo 7 minutos. Mostrar, não explicar.** O **Marcos Paulo fica no controle da tela o tempo
todo** (`python -m retrato_vivo.gui`), e quem está falando narra por cima com uma ou duas frases
de contexto. As explicações técnicas ficam para quando o professor perguntar
([seção no fim](#se-o-professor-perguntar)).

Meta: **6:20**, deixando 40 s de folga para interrupções.

| Tempo | Quem fala | O que o Marcos Paulo mostra |
|---|---|---|
| 0:00 – 0:30 | Marcos Paulo | `foto_risco3`: o divisor antes/depois |
| 0:30 – 1:30 | Igor | abas Etapas (mapa de danos) e Máscara |
| 1:30 – 2:45 | Marco | dispara o Retrato Vivo; cor, rostos e aba Bordas |
| 2:45 – 4:00 | Igor | Retrato Vivo pronto; Máquina do tempo com uma selfie |
| 4:00 – 4:40 | Marco | foto impressa tirada com o celular: perspectiva |
| 4:40 – 6:00 | Marcos Paulo | a foto que o professor trouxer |
| 6:00 – 6:20 | Marco | fechamento (tela parada no antes/depois da foto do professor) |

**Como funciona a dupla falar + clicar:**
- Quem fala **não aponta para a tela pedindo cliques**: o Marcos Paulo já sabe a sequência de cada
  bloco (está escrita abaixo, em "Marcos Paulo clica") e vai clicando no ritmo da fala.
- Cada bloco tem uma **deixa**, a frase que dispara o próximo clique. Quem fala usa a deixa, o
  Marcos Paulo clica, e a fala segue sobre a tela nova.
- Quem fala fica ao lado da tela do projetor, não atrás do notebook, olhando para o professor.
- Se o professor quiser mexer na interface, o Marcos Paulo passa o mouse para ele e conduz.

## Duas regras de ordem (por causa do tempo de processamento)

1. **Depois de clicar em "Criar Retrato Vivo", o Marcos Paulo não mexe nos controles do painel nem
   abre outra foto até o Retrato Vivo aparecer.** Qualquer controle refaz a restauração, e isso
   descarta o retrato em andamento. Trocar de aba pode.
2. **Começar com tudo pronto:** a `foto_risco3` aberta e restaurada com "Colorir" e "Melhorar
   rostos" ligados. Na CPU cada restauração com cor e rostos leva ~17 s, então nada de ligar e
   desligar controles ao vivo.

Tempos medidos num notebook **sem placa de vídeo**. Com NVIDIA fica de 5 a 10 vezes mais rápido.

| Ação | CPU |
|---|---|
| Restaurar (padrão) / com cor e rostos | ~8 s / ~17 s |
| **Retrato Vivo** | **~33 s** (na RTX 4080: 2 a 4 s) |
| Máquina do tempo: uma época / "Restaurar de volta" | < 1 s / ~4 a 8 s |
| Bordas Canny × HED | ~1 s |

---

## Antes da apresentação

**Na véspera (29/11):**
- [ ] Escolher o notebook. **Se der, o do Igor (RTX 4080):** o Retrato Vivo cai de ~33 s para
      2 a 4 s e as restaurações ficam instantâneas. A barra de status mostra se a IA está na GPU.
- [ ] No notebook: `git pull`, `pip install -r requirements.txt` e `python -m retrato_vivo modelos`
      (~1,9 GB). **Não deixar para a sala.**
- [ ] Abrir a interface e fazer o roteiro inteiro uma vez, para todos os modelos carregarem.
- [ ] **Plano B do Retrato Vivo:** salvar antes o vídeo da `foto_risco3`
      (`python -m retrato_vivo vivo imagens/foto_risco3.webp -o apresentacao`) e ter
      `docs/img/retrato_vivo.gif` à mão.
- [ ] Separar **2 ou 3 fotos antigas que ninguém usou no desenvolvimento**, caso o professor não traga.
- [ ] Levar **uma foto impressa** e já deixar no notebook uma foto dela tirada com o celular **sobre um
      fundo escuro** (papel claro sobre mesa clara quase nunca é achado). Se der tempo, fotografar
      de novo na hora.
- [ ] Ter uma selfie qualquer no notebook para a Máquina do tempo, caso o professor não queira tirar.
- [ ] **Marcos Paulo treina sozinho a sequência de cliques** (as linhas "Marcos Paulo clica"), até
      fazer sem olhar o roteiro.
- [ ] **Ensaiar duas vezes com cronômetro**, os três juntos, testando as deixas. Se passar de 6:30, cortar o que está marcado como
      "se sobrar tempo".

**30 minutos antes:**
- [ ] Notebook na tomada, modo "Melhor desempenho", notificações desligadas.
- [ ] Conectar no projetor e conferir que a janela cabe (largura mínima: 934 px).
- [ ] Interface aberta com a `foto_risco3` restaurada, "Colorir" e "Melhorar rostos" ligados.
- [ ] Terminal aberto na pasta do projeto (plano B).

---

## O roteiro

Em cada bloco: **Fala** é o que quem fala diz (em linhas gerais, não decorar), **Deixa** é a
frase que avisa o Marcos Paulo, e **Marcos Paulo clica** é a sequência na tela.

### 0:00 · Abertura (30 s) · fala: Marcos Paulo

- **Fala:** "Fotos de família se estragam: riscos, manchas, amarelado. O Retrato Vivo restaura
  automaticamente. Eu fico na tela e cada um vai apresentar a parte que fez."
- **Marcos Paulo clica:** na aba **Antes e depois**, arrasta o divisor devagar sobre a
  `foto_risco3`, de um lado ao outro e de volta ao meio.
- **Deixa para o Igor:** "Começando pela restauração, com o Igor."

### 0:30 · Restauração (1 min) · fala: Igor

- **Marcos Paulo clica:** aba **Etapas** → desce até o **mapa de danos** e para.
- **Fala:** "Cada etapa mede a foto e só age se ela precisar, e explica aqui o que decidiu. No mapa:
  riscos, vincos e manchas encontrados, e o rosto protegido em verde. Os riscos são achados por
  morfologia e por uma rede neural, e preenchidos por outra rede."
- **Deixa:** "...e aqui o resultado." → **Marcos Paulo** desce até **danos reparados**.
- **Deixa:** "E se a detecção deixar passar alguma coisa?" → **Marcos Paulo** vai na aba **Máscara**
  e pinta um traço com o pincel. **Não clica em "Aplicar"** (refaz a restauração e atrasa o
  Retrato Vivo).
- **Fala:** "O usuário marca à mão, e a restauração é refeita com a correção."
- **Deixa para o Marco:** "A parte de Deep Learning é com o Marco."

### 1:30 · Deep Learning (1 min 15 s) · fala: Marco

- **Marcos Paulo clica, logo que o Marco começa:** aba **Retrato Vivo** → **"Criar Retrato Vivo"**.
  A partir daqui, **nenhum controle do painel** até o bloco do Igor.
- **Fala, enquanto o Marcos Paulo clica:** "Já deixamos o diferencial do projeto processando; o Igor
  mostra daqui a pouco."
- **Deixa:** "A foto original era preto e branco." → **Marcos Paulo** vai na aba **Etapas** e mostra
  a etapa de **colorização** e depois a de **rostos**.
- **Fala:** "Uma rede previu as cores, e outra reconstruiu o rosto."
- **Deixa:** "Agora as bordas." → **Marcos Paulo** vai na aba **Bordas**.
- **Fala:** "Em cima, o método clássico; embaixo, uma rede neural, antes e depois. O clássico marca
  qualquer textura; a rede mostra os contornos reais, que ficam mais fortes depois da restauração.
  A interface também amplia fotos pequenas com super-resolução."
- **Deixa para o Igor:** "E o Retrato Vivo já deve estar pronto." → **Marcos Paulo** volta para a aba
  **Retrato Vivo**.

### 2:45 · Diferencial (1 min 15 s) · fala: Igor

- **Marcos Paulo clica:** nada; o vídeo já está tocando.
- **Fala:** "A foto ganha movimento 3D: uma rede estima a profundidade de cada pixel e uma câmera
  virtual se move."
- **Deixa:** "...a profundidade." → **Marcos Paulo** marca **"Ver profundidade"** e desmarca em seguida.
- **Fala:** "O contrário também: professor, podemos usar uma selfie sua?"
- **Marcos Paulo clica:** abre a selfie do professor (ou a de reserva) → aba **Máquina do tempo** →
  **1920**.
- **Deixa:** "E aí restauramos de volta." → **Marcos Paulo** clica em **"Restaurar de volta"**.
- **Fala, durante os ~5 s:** "Os riscos que a Máquina do tempo desenhou somem: é a prova ao vivo de
  que a restauração funciona."
- **Deixa para o Marco:** "E quem não tem scanner?"

### 4:00 · Foto de uma foto (40 s) · fala: Marco

- **Marcos Paulo clica:** marca **"Foto tirada do celular"** → abre a foto da foto impressa → aba
  **Perspectiva** (o papel contornado em verde) → aba **Antes e depois**, com a foto desentortada.
- **Fala:** "Quem não tem scanner fotografa a foto com o celular, e ela sai torta e com a mesa em
  volta. O sistema acha o papel, desentorta e recorta. Se errar um canto, dá para arrastar."
- **Deixa para o Marcos Paulo:** "Agora, professor, uma foto sua."

### 4:40 · A foto do professor (1 min 20 s) · fala: Marcos Paulo

- **Marcos Paulo clica:** desmarca **"Foto tirada do celular"** (se a foto do professor for
  digitalizada) e arrasta a foto para a janela.
- **Fala, enquanto restaura:** "A restauração roda em segundo plano, a janela não trava, e aceita
  qualquer formato."
- **Marcos Paulo clica:** o antes/depois → aponta a barra de status (danos encontrados, tempo, GPU ou
  CPU) → **"Exportar comparativo"**.
- Se a detecção errar algo: aba **Máscara** → pinta → **"Aplicar correções"**.
- **Deixa para o Marco:** "Para fechar, o Marco."

### 6:00 · Fechamento (20 s) · fala: Marco

- **Marcos Paulo clica:** volta para a aba **Antes e depois** da foto do professor e deixa parado.
- **Fala:** "Oito redes neurais, processamento de imagens clássico explicado na própria interface;
  roda em qualquer PC e usa a placa de vídeo quando tem. Limitações: rasgo grande precisa ser
  pintado à mão, e a super-resolução ajuda pouco. Perguntas?"

**Se sobrar tempo** (ensaio abaixo de 6 min): no bloco do Igor, o Marcos Paulo clica em "Aplicar"
na máscara *antes* de disparar o Retrato Vivo; no bloco do Marco, abre o painel "Ajustes avançados".

**Se faltar tempo:** cortar o "Ver profundidade" e a fala da super-resolução.

---

## Se o professor perguntar

Respostas curtas. Cada um responde pela própria parte.

### Igor: restauração, danos e diferencial

- **Diagnóstico adaptativo:** mede cor (desvio de a/b no LAB), ruído e faixa de tons; cada etapa
  só age se a foto precisar ("não estragar foto boa"). Sépia é detectada pelo tom dominante.
- **Riscos e vincos:** top-hat/black-hat com segmentos de reta em 12 direções, histerese e testes
  de forma; manchas por forma e contraste; rostos protegidos pelo YuNet.
- **IA nos danos:** detector de riscos da Microsoft (*Bringing Old Photos Back to Life*, CVPR 2020),
  57 riscos a mais na `foto_risco3` sem falso alarme na foto boa. Áreas grandes: **LaMa**.
- **Como sabem que melhora?** `python -m retrato_vivo avaliar`: estraga uma foto boa de propósito,
  restaura e mede. PSNR 22,6 → 30,7 dB na `foto_boa1`; detecção com F1 de 0,69.
- **Acabamento:** Non-Local Means (ruído), CLAHE (contraste local), nitidez.
- **Por que não só IA?** O clássico é rápido, explicável e não estraga foto boa; a IA entra onde
  as regras falham.
- **Retrato Vivo:** Depth Anything V2 (profundidade) → Otsu + morfologia separam a pessoa → LaMa
  inventa o fundo escondido → câmera virtual com `cv2.remap` (paralaxe) → MP4/GIF.
- **Máquina do tempo:** cada época é uma receita dos efeitos de `aging.py` (sépia, grão, vinheta,
  riscos, moldura).

### Marco: Deep Learning e perspectiva

- **Colorização (Zhang, ECCV 2016):** no espaço **LAB**, a rede vê só a luminosidade (L) e prevê
  a cor (a/b); junta com o L original, então nenhum detalhe se perde. **DDColor** (ICCV 2023) é o
  padrão, mais realista em 4 de 5 fotos. Só colore foto P&B ou sépia.
- **A cor é a verdadeira?** É a cor *provável*, aprendida em milhões de fotos; por isso há o ajuste
  de intensidade.
- **Super-resolução:** ESPCN (rápido) ou EDSR (melhor, lento); só amplia foto menor que 2048 px.
  Ganho medido de +0,2 a +1,1 dB sobre a ampliação comum: recupera nitidez de bordas, não inventa
  detalhe.
- **Canny × HED:** o Canny marca qualquer variação de brilho (mais traços depois, porque o CLAHE
  realça textura); o HED (Xie & Tu, ICCV 2015) marca só contornos de objetos, mais fortes depois.
- **Perspectiva:** Canny e Otsu → `approxPolyDP` (maior quadrilátero) → testes de desconfiança
  (convexo, fundo em volta, ângulos, lados apoiados em bordas) → **homografia**
  (`getPerspectiveTransform` + `warpPerspective`). Foto digitalizada volta intacta.
- **Por que OpenCV 4 e não 5?** O 5 removeu o leitor de Caffe (Zhang e HED). E o 4.14 tem um bug
  ao importar a camada final das redes de super-resolução, que corrigimos em memória (`ai/tf_fix.py`).

### Marcos Paulo: interface

- **PySide6 (Qt)**, a restauração numa `QThread`; mexer num controle refaz a restauração 0,4 s depois
  da última mexida, uma de cada vez.
- **Formatos:** JPG, PNG, WEBP, AVIF, 16 bits, transparência, acentos no nome, rotação de celular.
- **Máscara:** pincel e borracha, desfazer (Ctrl+Z), zoom até 8×; a pintura vira `mask_add` /
  `mask_remove` e a restauração é refeita.
- **Perspectiva errou?** Na aba Perspectiva o usuário arrasta os 4 cantos.
- **Roda em qualquer PC?** Sim: as redes pesadas usam a GPU (ONNX Runtime + CUDA) quando há
  NVIDIA e caem para a CPU sozinhas.
- **Testes:** mais de 300 testes automáticos (`python -m pytest`).

---

## Se algo der errado

| Problema | O que fazer |
|---|---|
| Retrato Vivo não ficou pronto | Fazer a Máquina do tempo antes e voltar a ele; ou abrir o MP4 salvo na véspera |
| Alguém mexeu num controle e o Retrato Vivo sumiu | Clicar em "Criar" de novo e seguir para a Máquina do tempo |
| A interface travou ou fechou | `python -m retrato_vivo.gui` (abre sozinha com as fotos de `imagens/`) |
| A foto do professor ficou ruim | Mostrar na aba Etapas o que foi decidido e corrigir na aba Máscara |
| A perspectiva não achou o papel | Arrastar os 4 cantos na aba Perspectiva (ponto a favor, não falha) |
