# SPEC-006: Privacidade e exclusão de pessoa

## Local dos dados

Todos os dados privados ficam em `FACEREC_HOME` (padrão: `%LOCALAPPDATA%\facerec-ufms`
no Windows e `~/.local/share/facerec-ufms` nos demais sistemas). O padrão fica fora
do repositório e fora de pastas sincronizadas.

A aplicação **recusa** iniciar se `FACEREC_HOME` estiver dentro de um repositório git
ou de uma pasta sincronizada (OneDrive, Dropbox, Google Drive, iCloud).

## Exclusão

A exclusão remove, em uma única operação: cadastro, embeddings, fotografias (se
houver) e demais dados associados. O SQLite roda com `secure_delete=ON` e faz `VACUUM`
depois da exclusão, para que os bytes não permaneçam no arquivo.

Também existe `scripts/purge_all.py`, para a eliminação de todos os dados ao final da
disciplina (§24 do enunciado).

## Logs

Logs registram apenas evento, identificador opaco e código de erro. Nunca nome,
imagem, embedding ou similaridade individual. Ficam em `FACEREC_HOME/logs`.

## Critérios de aceitação

- **CA-006.1** Após excluir, não restam linhas da pessoa em nenhuma tabela.
- **CA-006.2** Após excluir, as fotografias da pessoa não existem no disco.
- **CA-006.3** Após excluir, a pessoa não é mais reconhecida.
- **CA-006.4** Após excluir, o nome e os bytes do embedding não aparecem no arquivo do banco.
- **CA-006.5** `FACEREC_HOME` dentro de repositório git ou pasta sincronizada impede a inicialização.
- **CA-006.6** O `.gitignore` ignora os caminhos privados do enunciado (§18 e §19).
- **CA-006.7** Excluir pessoa inexistente retorna 404.
