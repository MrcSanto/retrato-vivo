# AGENTS.md

Instruções para qualquer agente de IA (Claude, Codex, Gemini, Cursor...) que trabalhe
neste repositório. Fonte única: o `CLAUDE.md` apenas importa este arquivo.

## Projeto

**Retrato Vivo**: restaurador de fotos antigas com OpenCV e Deep Learning. Trabalho
final de CCC309 – Processamento de Imagens e Visão Computacional (UPF, 2026/2).
Apresentação em **30/11/2026**.

Antes de mudar algo, leia:

- `docs/FLUXO_DE_TRABALHO.md`: arquitetura, contrato entre módulos, divisão de tarefas,
  prazos e regras do repositório.
- `docs/COMO_FUNCIONA.md`: o que cada etapa da restauração clássica faz.

## Responsáveis por área

| Pasta / arquivo | Responsável |
|---|---|
| `io_utils`, `preprocess`, `faces`, `damage`, `inpaint`, `visualize`, `aging`, `metrics` | Igor Zanette |
| `retrato_vivo/ai/` (colorização, super-resolução, rosto, profundidade) | Marco Antonio Santolin |
| `retrato_vivo/gui/` (PySide6) | Marcos Paulo de Medeiros |
| `pipeline.py`, `models.py` | todos (ponto de integração) |

Mexa só na área da pessoa que está pedindo. Se precisar alterar código de outra área,
avise antes em vez de editar.

## Comandos

```bash
pip install -r requirements.txt
python -m retrato_vivo modelos                                   # baixa os pesos para models/
python -m pytest                                                 # precisa passar sempre
python -m retrato_vivo restaurar foto.jpg -o saida --painel --explicar
python -m retrato_vivo avaliar foto_boa.jpg -o avaliacao         # PSNR/SSIM
```

## Contrato de código

- Imagem é sempre `np.ndarray` **BGR `uint8` `(H, W, 3)`**.
- Abra com `io_utils.load_image` e salve com `io_utils.save_image`. **Nunca** use
  `cv2.imread`/`cv2.imwrite` direto (quebra com acentos no Windows e ignora a rotação EXIF).
- Etapa de IA (*post stage*): `def etapa(img: np.ndarray, result: RestorationResult) -> np.ndarray`,
  sem estado global escondido. Pode ler `result.diagnosis` e `result.damage.faces`.
- Inpainter alternativo: `def inpainter(img, mask_uint8) -> np.ndarray`.
- A GUI só conversa com `pipeline.restore()`.
- Modelos pré-treinados: registre em `retrato_vivo/models.py` e use
  `models.model_path("nome")`. Carregue com `cv2.dnn` (não há `opencv-contrib` nem CUDA).
- Toda função nova ganha pelo menos um teste em `tests/`. Testes que dependem de modelo
  baixado usam `pytest.mark.skipif`.
- Código, comentários, docstrings e textos de interface em **português**.

## Regras do repositório (avaliadas pelo professor)

- Cada pessoa commita o próprio trabalho com a própria conta. O agente não faz commit
  nem push sem pedido explícito.
- Uma branch por tarefa (`marco/colorizacao`, `igor/vincos`, `marcos/gui-mascara`),
  integrada na `main` por Pull Request.
- Commits pequenos, mensagem em português no imperativo: `Adiciona colorização`.
- `python -m pytest` precisa passar antes de abrir o PR.
- **Nunca** versionar: pesos (`models/`), fotos pessoais, `saida/`, `avaliacao/`, `.venv/`.
- Ao concluir algo, atualize a seção "Status" da área em `docs/FLUXO_DE_TRABALHO.md`.
