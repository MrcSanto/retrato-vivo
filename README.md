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
- Usa **IA para achar riscos** (rede da Microsoft treinada em fotos antigas) e para
  **reconstruir áreas grandes** como rasgos (LaMa).
- **Colore** fotos preto e branco (DDColor, ICCV 2023, ou Zhang, ECCV 2016) e
  **restaura rostos** (GFPGAN).
- Roda as redes na **GPU (CUDA)** quando houver placa NVIDIA, e na CPU em qualquer PC.
- Tem **interface gráfica**: comparação antes e depois, a foto em cada etapa, ajustes da
  restauração e um editor para corrigir a detecção à mão.
- **Retrato Vivo** (o diferencial): a foto restaurada vira um vídeo com efeito 3D.
- **Máquina do tempo**: uma foto atual vira foto de 1900, 1920, 1950, 1970 ou Polaroid.
- _Em desenvolvimento:_ super-resolução com Deep Learning.

## Como usar

```bash
pip install -r requirements.txt
python -m retrato_vivo modelos                       # baixa os modelos pré-treinados (~1,8 GB)
pip install "onnxruntime-gpu[cuda,cudnn]"            # opcional: usar placa NVIDIA
python -m retrato_vivo.gui [foto.jpg]                # interface gráfica
python -m retrato_vivo restaurar foto.jpg -o saida --painel --explicar
python -m retrato_vivo avaliar foto_boa.jpg          # estraga, restaura e mede PSNR/SSIM
python -m retrato_vivo vivo foto.jpg -o saida        # Retrato Vivo: vídeo 3D (MP4 e GIF)
python -m retrato_vivo viajar selfie.jpg -o saida     # Máquina do tempo: todas as épocas
python -m pytest
```

## Interface gráfica

Abra com `python -m retrato_vivo.gui`. Sem argumentos, já abre a primeira foto de `imagens/`
(fotos de exemplo); ◀ e ▶ passam pelas outras fotos da pasta.

1. **Abrir:** clique em "Abrir" (Ctrl+O) ou arraste a foto para a janela. A restauração começa sozinha.
2. **Antes e depois:** arraste a linha branca sobre a foto.
3. **Painel da esquerda:** "Força da limpeza", "Colorir" e "Melhorar rostos". O resto fica em
   "Ajustes avançados". A restauração é refeita sozinha quando você para de mexer.
4. **Etapas:** a foto em cada ponto do processamento e o que cada etapa decidiu.
5. **Máscara:** pinte com o pincel o que a detecção não pegou (rasgos, por exemplo) e clique em
   "Aplicar correções". A borracha desmarca o que foi marcado por engano.
6. **Retrato Vivo:** "Criar Retrato Vivo" transforma a foto restaurada num vídeo 3D; salve em MP4 ou GIF.
7. **Máquina do tempo:** escolha uma época; "Restaurar de volta" mostra a restauração desfazendo o
   envelhecimento.
8. **Salvar:** "Salvar resultado" (Ctrl+S) e "Exportar comparativo".

A barra de status mostra os danos encontrados, o tempo e se a IA está rodando na GPU ou na CPU.

## Documentação

- **[docs/COMO_FUNCIONA.md](docs/COMO_FUNCIONA.md)**: o que cada arquivo e cada etapa faz.
- **[docs/DIFERENCIAL.md](docs/DIFERENCIAL.md)**: Retrato Vivo e Máquina do tempo, e o roteiro da apresentação.
- **[docs/FLUXO_DE_TRABALHO.md](docs/FLUXO_DE_TRABALHO.md)**: divisão de tarefas, prazos e regras para contribuir.
