# ADR-002: Armazenamento de embeddings

**Status:** aceito

## Contexto

A galeria é pequena, de dezenas de pessoas, e guarda dados biométricos que precisam
ser removidos por pessoa, de forma completa (§24 e §39 do enunciado).

## Decisão

SQLite local em `FACEREC_HOME/gallery/faces.db`. Cada embedding fica em uma linha
ligada à pessoa com `ON DELETE CASCADE`. Busca linear em memória com NumPy.
Fotografias não são guardadas por padrão.

`secure_delete=ON`, `VACUUM` depois da exclusão e `journal_mode=DELETE` evitam que
bytes apagados continuem no arquivo ou em um WAL.

## Alternativas

- **FAISS ou banco vetorial**: desnecessário nessa escala e dificulta a exclusão garantida.
- **Arquivos `.npy` por pessoa**: exclusão simples, mas sem atomicidade no cadastro.
- **Guardar fotos**: permitiria regerar embeddings ao trocar de modelo, porém contraria a minimização de dados.

## Consequências

Trocar o modelo exige recadastrar as pessoas, já que cada embedding registra o modelo
que o gerou e os demais são ignorados, com aviso na interface. O arquivo do banco não
é cifrado: a proteção depende do controle de acesso da conta do sistema operacional.
