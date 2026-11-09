# ADR-001: Modelo de reconhecimento

**Status:** aceito. Substitui a decisão anterior, que usava E05, depois da inclusão do CelebA (SPEC-008).

## Contexto

Foram 10 experimentos no LFW (E01 a E10) e 6 avaliados em LFW e CelebA (C01 a C06). A
regra final, definida antes dos treinos no CelebA, escolhe a menor **média do EER de
validação nos dois datasets**.

## Decisão

Usar **C01**: FaceNet (Inception-ResNet v1, 512 dimensões) com os pesos públicos
VGGFace2, **sem fine-tuning**. Detecção e alinhamento com MTCNN. Comparação com o
**protótipo** da pessoa (média normalizada dos embeddings), que teve a maior taxa de
acerto na validação nos dois datasets entre os comparadores viáveis.

## Justificativa

| Exp. | Configuração | Média EER val | EER val CelebA | EER val LFW |
|---|---|---|---|---|
| **C01** | **Sem fine-tuning** | **0,0239** | 0,0399 | 0,0079 |
| C05 | Parcial, Triplet, CelebA | 0,0245 | 0,0403 | 0,0087 |
| C02 | E05 (parcial, softmax, LFW) | 0,0250 | 0,0437 | 0,0064 |
| C04 | Parcial, ArcFace, CelebA | 0,0252 | 0,0416 | 0,0087 |
| C06 | Completo, ArcFace, CelebA | 0,0252 | 0,0415 | 0,0090 |
| C03 | Parcial, softmax, CelebA | 0,0265 | 0,0422 | 0,0108 |

As diferenças são pequenas e os intervalos de confiança se sobrepõem. Portanto, o
resultado **não prova** que o pré-treinado é superior; mostra que nenhum fine-tuning
testado trouxe ganho que justificasse seu custo e sua complexidade. A escolha do
pré-treinado também dispensa checkpoint local: quem clonar o projeto só precisa dos
pesos públicos.

O E05, escolhido apenas com o LFW, tinha o melhor EER nesse dataset, mas ficou pior no
CelebA (0,0437 contra 0,0399). A vantagem era, ao menos em parte, específica do LFW.

## Alternativas

- **E05 ou C02**: melhor no LFW, pior no CelebA.
- **C06 (fine-tuning completo)**: 2.027 s de treino, sem ganho.
- **SVM como comparador**: exige retreino a cada cadastro.

## Consequências

Se o grupo fizer fine-tuning com as fotos privadas, o checkpoint resultante passa a ser
privado (§19). Checkpoints derivados do CelebA não são publicados, conforme os termos
de uso desse dataset. Trocar o modelo exige recadastrar as pessoas.
