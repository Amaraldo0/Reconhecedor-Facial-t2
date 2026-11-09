# SPEC-003: Reconhecimento

## Fluxo

Imagem ou quadro de câmera → Detecção → Embedding → Comparação → Threshold →
Reconhecido ou desconhecido.

Todas as faces da imagem são processadas e cada uma recebe sua própria decisão. Sem
pessoas cadastradas, toda face é desconhecida. Os embeddings são comparados por
similaridade de cosseno, e o modelo e o método de agregação vêm da configuração
escolhida na Parte 1 (ADR-001).

Cada embedding é gravado com o identificador do modelo que o gerou. Embeddings de
modelos diferentes não são comparáveis.

## Critérios de aceitação

- **CA-003.1** A face de uma pessoa cadastrada é reconhecida com o nome correto e a similaridade.
- **CA-003.2** Imagem com várias faces retorna uma decisão por face.
- **CA-003.3** Imagem sem faces retorna lista vazia e não é erro.
- **CA-003.4** A galeria inicia com o modelo configurado; se houver embeddings de outro modelo, a aplicação avisa e não os compara.
- **CA-003.5** O reconhecimento não grava imagem nem embedding da face consultada.
- **CA-003.6** Quando a galeria muda, a próxima consulta já reflete a mudança.
