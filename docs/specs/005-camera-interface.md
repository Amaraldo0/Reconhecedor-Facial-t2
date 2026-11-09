# SPEC-005: Interface web local com câmera

## Comportamento

Servidor FastAPI, ligado apenas a `127.0.0.1`, serve uma página sem dependências
externas (nenhum CDN). Abas:

- **Reconhecer**: vídeo da câmera com caixas e rótulos sobre as faces, ou envio de imagem.
- **Cadastrar**: nome, confirmação explícita de consentimento e fotos enviadas ou capturadas.
- **Pessoas**: lista de cadastrados e exclusão com confirmação.

Os quadros da câmera ficam em memória durante o processamento e não são gravados.
Só um quadro é processado por vez, para que a interface não acumule atraso.

## Critérios de aceitação

- **CA-005.1** Galeria vazia mostra "Nenhuma pessoa cadastrada".
- **CA-005.2** O servidor escuta somente em loopback por padrão.
- **CA-005.3** Requisições com `Host` ou `Origin` externos são recusadas.
- **CA-005.4** Reconhecido exibe nome e caixa verde; desconhecido exibe "Pessoa desconhecida" e caixa vermelha.
- **CA-005.5** Erros de cadastro aparecem em português e explicam como corrigir.
- **CA-005.6** Câmera indisponível ou negada mostra aviso e mantém o envio de imagem funcionando.
- **CA-005.7** A página não carrega nenhum recurso externo.
