# ADR-003: Threshold de reconhecimento

**Status:** aceito, a reavaliar com o dataset privado

## Contexto

O threshold define o equilíbrio entre aceitar um estranho e rejeitar um cadastrado. A
taxa de falsa aceitação depende do tamanho da galeria: quanto mais pessoas, maior a
chance de uma face desconhecida se parecer com alguém.

## Decisão

**`recognition.threshold = 0.673`**, com o modelo C01 e o comparador por protótipo.

Calibrado na validação para galeria de 20 pessoas e FAR-alvo de 0,1%, adotando o maior
valor entre LFW (0,673) e CelebA (0,647), como definido na SPEC-008.

Neste sistema, nomear um estranho é pior que pedir nova tentativa. Em vídeo, uma
rejeição se corrige no quadro seguinte, enquanto uma falsa aceitação identifica a
pessoa errada. Por isso o alvo é 0,1%, e não 1%.

## Evidência

Threshold da validação, medido no teste do mesmo dataset, com 30 sorteios de galeria:

| Dataset | Pessoas | FAR-alvo | Threshold | FAR no teste | Identificação correta |
|---|---|---|---|---|---|
| LFW | 20 | 1% | 0,579 | 0,77% | 97,6% |
| **LFW** | **20** | **0,1%** | **0,673** | **0,12%** | **93,4%** |
| CelebA | 20 | 1% | 0,557 | 1,17% | 94,6% |
| CelebA | 20 | 0,1% | 0,647 | 0,13% | 87,7% |

No CelebA, o valor adotado (0,673) é mais rigoroso que o calibrado para ele, então a
identificação correta tende a ficar abaixo de 87,7%. Galerias de 5 e 50 pessoas estão em
`experiments/results/SUMMARY_CELEBA.md`.

O threshold de EER em pares não foi adotado, pois trata cada comparação isoladamente e
subestima o risco de comparar uma face com toda a galeria.

## Consequências

Os valores vêm de fotos de celebridades, diferentes de uma webcam. Cadastrar várias
fotos por pessoa reduz as rejeições. O grupo deve rodar
`experiments/evaluate_private.py` e rever o valor, sem usar o teste privado para isso se
ele for reportado como resultado final.
