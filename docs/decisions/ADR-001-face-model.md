# ADR-001: Modelo de reconhecimento

**Status:** aceito

## Contexto

A Parte 1 comparou dez configurações no LFW (ver `experiments/results/SUMMARY.md`). A
regra definida antes da execução foi escolher o menor EER no conjunto de **validação**.

## Decisão

Usar **E05**: FaceNet (Inception-ResNet v1, 512 dimensões) pré-treinado em VGGFace2,
com fine-tuning parcial (`repeat_3` em diante, 8,9 M de 27,9 M de parâmetros) e
entropia cruzada sobre similaridade de cosseno. Detecção e alinhamento com MTCNN.

Na identificação, comparar com o **protótipo** (média normalizada dos embeddings da
pessoa), que teve a maior taxa de acerto na validação entre os comparadores que não
exigem retreino a cada cadastro (0,9562, contra 0,9545 da melhor amostra individual).

## Justificativa e ressalva

| | EER validação | EER teste (IC 95%) |
|---|---|---|
| E01 (sem fine-tuning) | 0,0079 | 0,0075 (0,0047–0,0118) |
| E05 (escolhido) | 0,0064 | 0,0070 (0,0036–0,0106) |

E05 venceu pela regra, mas **a vantagem sobre E01 está dentro do ruído**: os intervalos
se sobrepõem bastante. O fine-tuning parcial custou cerca de 21 s de GPU, e E05 teve o
menor EER no teste degradado (0,0172, contra 0,0221 de E01). O LFW está perto do teto
de desempenho desses modelos, o que limita o que se pode concluir sobre o fine-tuning.

## Alternativas

- **E01**: mais simples e sem checkpoint local. Para usá-lo, defina `model.id: facenet-vggface2` e `model.checkpoint: null`, e recadastre as pessoas.
- **E06 (Triplet)**: maior accuracy no teste, mas não pode ser escolhido por isso; na validação ficou atrás de E05.
- **E07 (fine-tuning completo)**: sem ganho e cerca de 19 vezes mais caro.
- **SVM como comparador**: exigiria retreino a cada cadastro e não superou o protótipo na validação.

## Consequências

O checkpoint é gerado localmente e não é versionado. Como foi treinado só com dados
públicos, não é biométrico dos participantes. Se o grupo fizer fine-tuning com fotos
privadas, o novo checkpoint passa a ser privado (§19).
