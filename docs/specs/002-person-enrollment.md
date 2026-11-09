# SPEC-002: Cadastro de pessoa

## Fluxo

Pessoa → Imagem(ns) → Detecção → Alinhamento → Embedding → Cadastro.

A requisição traz nome, confirmação de consentimento e de 1 a N imagens (N
configurável). O cadastro é **atômico**: se uma imagem falhar, nada é gravado e a
resposta indica qual imagem falhou e por quê.

Por minimização de dados, o padrão é **não guardar fotografias**, apenas embeddings
(`storage.store_photos: false`).

## Regras

| Situação | Erro | HTTP |
|----------|------|------|
| Consentimento não confirmado | `ConsentRequiredError` | 400 |
| Nome vazio ou inválido | `InvalidNameError` | 400 |
| Arquivo inválido | `InvalidImageError` | 400 |
| Nenhuma face | `NoFaceError` | 422 |
| Mais de uma face | `MultipleFacesError` | 422 |
| Face pequena, borrada, escura, estourada ou de baixa confiança | `InadequateImageError` | 422 |
| Nome já cadastrado (ignora maiúsculas e acentos) | `DuplicatePersonError` | 409 |
| Face já cadastrada sob outro nome | `DuplicateFaceError` | 409 |
| Imagens do mesmo envio que parecem ser de pessoas diferentes | `InconsistentIdentityError` | 422 |

## Critérios de aceitação

- **CA-002.1** Sem consentimento confirmado, nada é processado nem gravado.
- **CA-002.2** Imagem sem face é rejeitada com `NoFaceError`.
- **CA-002.3** Imagem com duas faces é rejeitada com `MultipleFacesError`.
- **CA-002.4** Face menor que o mínimo, borrada, escura ou estourada é rejeitada com `InadequateImageError`, informando o motivo.
- **CA-002.5** Nome repetido é rejeitado, inclusive com variação de caixa e acento.
- **CA-002.6** Face que corresponde a outra pessoa cadastrada é rejeitada.
- **CA-002.7** Falha em qualquer imagem deixa o banco inalterado.
- **CA-002.8** Com `store_photos: false`, nenhum arquivo de imagem é gravado.
- **CA-002.9** Imagens de um mesmo envio com identidades divergentes são rejeitadas.
- **CA-002.10** Logs não contêm o nome da pessoa.
