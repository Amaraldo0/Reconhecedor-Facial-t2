# SPEC-008: CelebA como segundo dataset

## Motivação

No LFW, os modelos com VGGFace2 ficaram estatisticamente empatados, perto do teto das
métricas, e o treino tinha só 5.737 imagens. O CelebA oferece muito mais identidades
para treinar e uma segunda referência de avaliação.

## Dados

- Imagens `img_align_celeba`, processadas pelo mesmo MTCNN e pelo mesmo alinhamento da
  aplicação, pois o modelo precisa receber recortes iguais aos de produção.
- Identidades de `identity_CelebA.txt`, conferido por MD5.
- Partição oficial (`list_eval_partition`), que separa as identidades.
- Até 10 imagens por identidade (minimização e custo de disco). Seleção determinística.

## Vazamento entre datasets

LFW e CelebA contêm celebridades e o CelebA não publica nomes, portanto a mesma pessoa
pode estar nos dois. Para que o teste do LFW continue independente, remove-se do treino
do CelebA toda identidade cujo centróide, medido com o modelo pré-treinado E01, tenha
similaridade de cosseno >= 0,60 com o centróide de alguma identidade da validação ou do
teste do LFW. O valor é propositalmente baixo: uma remoção indevida só custa uma
identidade de treino. O total removido é relatado.

Os experimentos que treinaram no LFW (E05) não têm essa garantia em relação ao teste do
CelebA, e isso é relatado como limitação.

## Experimentos

Reavaliação de E01 e E05 e fine-tuning no CelebA, com a mesma avaliação nos dois
datasets. Definições em `configs/experiments_celeba.yaml`.

## Regra de escolha (definida antes da execução)

Menor **média do EER de validação no LFW e no CelebA**. Média simples, para nenhum
dataset dominar. Empate é decidido pelo menor custo de treino.

Threshold da aplicação: calibrado para galeria de 20 pessoas e FAR-alvo de 0,1% na
validação de cada dataset, adotando o **maior** dos dois valores, pois aceitar um
estranho é o erro mais grave.

## Critérios de aceitação

- **CA-008.1** A preparação falha se o MD5 do arquivo de identidades divergir.
- **CA-008.2** A preparação falha se uma identidade cruzar as partições oficiais.
- **CA-008.3** O limite de imagens por identidade é respeitado e determinístico.
- **CA-008.4** Identidades com centróide acima do limiar são removidas do treino, e as demais são mantidas.
- **CA-008.5** Checkpoints e recortes ficam em `FACEREC_HOME`, nunca no repositório.
