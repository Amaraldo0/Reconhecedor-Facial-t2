# Resultados agregados (gerado por experiments/summarize.py)

Melhor por EER de validação: **E05**.

| Experimento | Modelo | Estratégia | Perda | Dataset | Accuracy | F1 | AUC | EER val | EER teste (IC 95%) | FAR | FRR | EER degradado | Tempo (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E01 | FaceNet VGGFace2 | sem fine-tuning | none | LFW | 0.9919 | 0.9919 | 0.9998 | 0.0079 | 0.0075 (0.0047–0.0118) | 0.0091 | 0.0070 | 0.0221 | - |
| E02 | FaceNet CASIA | sem fine-tuning | none | LFW | 0.9863 | 0.9863 | 0.9991 | 0.0167 | 0.0145 (0.0094–0.0186) | 0.0124 | 0.0151 | 0.0312 | - |
| E03 | FaceNet VGGFace2 | sem fine-tuning | none | LFW | 0.9925 | 0.9925 | 0.9998 | 0.0079 | 0.0081 (0.0036–0.0115) | 0.0081 | 0.0070 | 0.0188 | - |
| E04 | FaceNet VGGFace2 | fine-tuning parcial | arcface | LFW | 0.9914 | 0.9914 | 0.9997 | 0.0069 | 0.0081 (0.0046–0.0127) | 0.0097 | 0.0075 | 0.0196 | 20 |
| E05 | FaceNet VGGFace2 | fine-tuning parcial | crossentropy | LFW | 0.9919 | 0.9920 | 0.9998 | 0.0064 | 0.0070 (0.0036–0.0106) | 0.0118 | 0.0043 | 0.0172 | 21 |
| E06 | FaceNet VGGFace2 | fine-tuning parcial | triplet | LFW | 0.9930 | 0.9930 | 0.9999 | 0.0069 | 0.0070 (0.0042–0.0106) | 0.0091 | 0.0048 | 0.0186 | 16 |
| E07 | FaceNet VGGFace2 | fine-tuning completo | arcface | LFW | 0.9925 | 0.9925 | 0.9998 | 0.0072 | 0.0078 (0.0044–0.0113) | 0.0124 | 0.0027 | 0.0204 | 401 |
| E08 | FaceNet CASIA | fine-tuning parcial | arcface | LFW | 0.9917 | 0.9917 | 0.9995 | 0.0138 | 0.0091 (0.0055–0.0120) | 0.0070 | 0.0097 | 0.0269 | 24 |
| E09 | Inception-ResNet | do zero | arcface | LFW | 0.7579 | 0.7572 | 0.8389 | 0.2396 | 0.2431 (0.2223–0.2645) | 0.2388 | 0.2453 | 0.3354 | 499 |
| E10 | FaceNet VGGFace2 | fine-tuning parcial | arcface | LFW | 0.9914 | 0.9914 | 0.9998 | 0.0067 | 0.0083 (0.0044–0.0115) | 0.0124 | 0.0048 | 0.0188 | 22 |

## Comparadores

Modelo: E05. FAR-alvo de 1% definido na validação.

| Comparador | Identificação correta (validação) | Identificação correta (teste) | FAR entre desconhecidos (teste) | Identidade errada aceita (teste) | Threshold |
|---|---|---|---|---|---|
| cosine_max | 0.9545 | 0.9023 | 0.0094 | 0.0000 | 0.653 |
| euclidean | 0.9545 | 0.9023 | 0.0094 | 0.0000 | -0.833 |
| prototype | 0.9562 | 0.9316 | 0.0094 | 0.0000 | 0.641 |
| knn3 | 0.7846 | 0.7850 | 0.0074 | 0.0000 | 0.546 |
| svm | 0.9440 | 0.9283 | 0.0087 | 0.0000 | -0.288 |

## Threshold por tamanho de galeria (prototype)

Threshold calibrado na validação e medido no teste, com 30 sorteios de galeria.

| Pessoas na galeria | FAR-alvo | Threshold | FAR no teste | Identificação correta no teste |
|---|---|---|---|---|
| 5 | 0.010 | 0.485 | 0.0068 | 0.9757 |
| 5 | 0.001 | 0.613 | 0.0006 | 0.9271 |
| 20 | 0.010 | 0.553 | 0.0075 | 0.9696 |
| 20 | 0.001 | 0.642 | 0.0013 | 0.9106 |
| 50 | 0.010 | 0.600 | 0.0073 | 0.9642 |
| 50 | 0.001 | 0.685 | 0.0015 | 0.8840 |

## Controle de vazamento

```json
{
  "removed_from_val": 5,
  "removed_from_train": 18,
  "train_images": 5737,
  "train_identities": 1008,
  "val_images": 2613,
  "val_identities": 1148,
  "test_images": 2424,
  "test_identities": 1150
}
```

## Thresholds do melhor modelo (validação)

```json
{
  "threshold_eer": 0.39139705896377563,
  "threshold_far1": 0.3612801730632782,
  "threshold_open_far1": 0.6530550122261047
}
```