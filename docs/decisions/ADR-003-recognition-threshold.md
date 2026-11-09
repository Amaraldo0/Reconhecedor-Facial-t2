# ADR-003: Threshold de reconhecimento

**Status:** aceito, a reavaliar com o dataset privado

## Contexto

O threshold define o equilíbrio entre aceitar um estranho e rejeitar um cadastrado. A
taxa de falsa aceitação depende do tamanho da galeria: quanto mais pessoas, maior a
chance de uma face desconhecida se parecer com alguém.

## Decisão

**`recognition.threshold = 0.642`**, calibrado na validação para galeria de 20 pessoas
e FAR-alvo de 0,1%, com o comparador por protótipo.

Neste sistema, nomear um estranho é pior que pedir nova tentativa. Em vídeo, uma
rejeição se corrige no quadro seguinte, enquanto uma falsa aceitação identifica a
pessoa errada. Por isso o alvo é 0,1%, e não 1%.

## Evidência

Threshold da validação, medido no teste, com 30 sorteios de galeria:

| Pessoas | FAR-alvo | Threshold | FAR no teste | Identificação correta |
|---|---|---|---|---|
| 5 | 1% | 0,485 | 0,68% | 97,6% |
| 5 | 0,1% | 0,613 | 0,06% | 92,7% |
| 20 | 1% | 0,553 | 0,75% | 97,0% |
| **20** | **0,1%** | **0,642** | **0,13%** | **91,1%** |
| 50 | 1% | 0,600 | 0,73% | 96,4% |
| 50 | 0,1% | 0,685 | 0,15% | 88,4% |

O threshold de EER em pares (0,391) não foi adotado, pois trata cada comparação
isoladamente e subestima o risco de comparar uma face com toda a galeria.

## Consequências

Os valores vêm do LFW, cujas fotos diferem de uma webcam. O grupo deve rodar
`experiments/evaluate_private.py` e rever o valor, sem usar o teste privado para isso
se ele for reportado como resultado final.
