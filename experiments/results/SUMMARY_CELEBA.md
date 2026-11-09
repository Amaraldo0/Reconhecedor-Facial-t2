# Resultados com CelebA (gerado por experiments/summarize_celeba.py)

Melhor pela média do EER de validação: **C01**. Comparador: **prototype**. Limiar da aplicação: **0.673**.

| Exp. | Configuração | Estratégia | Perda | Score (média EER val) | EER val CelebA | EER val LFW | EER teste CelebA (IC 95%) | Acc. CelebA | EER teste LFW (IC 95%) | Acc. LFW | EER degradado CelebA | Tempo (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | E01 reavaliado (sem fine-tuning) | sem fine-tuning | none | 0.0239 | 0.0399 | 0.0079 | 0.0369 (0.0331–0.0419) | 0.9625 | 0.0075 (0.0047–0.0118) | 0.9919 | 0.0534 | - |
| C02 | E05 reavaliado (fine-tuning no LFW) | sem fine-tuning | none | 0.0250 | 0.0437 | 0.0064 | 0.0394 (0.0356–0.0437) | 0.9600 | 0.0070 (0.0036–0.0106) | 0.9919 | 0.0601 | - |
| C03 | Fine-tuning parcial no CelebA, softmax | fine-tuning parcial | crossentropy | 0.0265 | 0.0422 | 0.0108 | 0.0407 (0.0364–0.0454) | 0.9571 | 0.0086 (0.0041–0.0118) | 0.9919 | 0.0598 | 108 |
| C04 | Fine-tuning parcial no CelebA, ArcFace | fine-tuning parcial | arcface | 0.0252 | 0.0416 | 0.0087 | 0.0388 (0.0346–0.0442) | 0.9596 | 0.0073 (0.0038–0.0111) | 0.9927 | 0.0568 | 200 |
| C05 | Fine-tuning parcial no CelebA, Triplet | fine-tuning parcial | triplet | 0.0245 | 0.0403 | 0.0087 | 0.0376 (0.0335–0.0423) | 0.9608 | 0.0083 (0.0048–0.0131) | 0.9914 | 0.0548 | 103 |
| C06 | Fine-tuning completo no CelebA, ArcFace | fine-tuning completo | arcface | 0.0252 | 0.0415 | 0.0090 | 0.0380 (0.0335–0.0429) | 0.9599 | 0.0070 (0.0036–0.0105) | 0.9927 | 0.0565 | 2027 |

## Controle de vazamento

```json
{
  "entre_particoes": {
    "removed_from_val": 218,
    "removed_from_train": 2545
  },
  "sobreposicao_com_lfw": {
    "threshold": 0.6,
    "identities_removed": 1792,
    "identities_kept": 6399,
    "images_kept": 56742
  }
}
```

## Comparadores (melhor modelo)

| Dataset | Comparador | Identificação correta (validação) | Identificação correta (teste) | FAR desconhecidos (teste) |
|---|---|---|---|---|
| lfw | cosine_max | 0.9335 | 0.8990 | 0.0087 |
| lfw | prototype | 0.9457 | 0.9235 | 0.0074 |
| lfw | svm | 0.9510 | 0.9202 | 0.0107 |
| celeba | cosine_max | 0.7880 | 0.7984 | 0.0154 |
| celeba | prototype | 0.8216 | 0.8283 | 0.0154 |
| celeba | svm | 0.8026 | 0.8053 | 0.0147 |

## Limiar por tamanho de galeria

| Dataset | Pessoas | FAR-alvo | Limiar (validação) | FAR no teste | Identificação correta no teste |
|---|---|---|---|---|---|
| lfw | 5 | 0.010 | 0.513 | 0.0078 | 0.9863 |
| lfw | 5 | 0.001 | 0.655 | 0.0002 | 0.9514 |
| lfw | 20 | 0.010 | 0.579 | 0.0077 | 0.9760 |
| lfw | 20 | 0.001 | 0.673 | 0.0012 | 0.9338 |
| lfw | 50 | 0.010 | 0.632 | 0.0069 | 0.9649 |
| lfw | 50 | 0.001 | 0.718 | 0.0011 | 0.8885 |
| celeba | 5 | 0.010 | 0.482 | 0.0114 | 0.9564 |
| celeba | 5 | 0.001 | 0.589 | 0.0015 | 0.9203 |
| celeba | 20 | 0.010 | 0.557 | 0.0117 | 0.9459 |
| celeba | 20 | 0.001 | 0.647 | 0.0013 | 0.8765 |
| celeba | 50 | 0.010 | 0.596 | 0.0133 | 0.9230 |
| celeba | 50 | 0.001 | 0.678 | 0.0018 | 0.8477 |