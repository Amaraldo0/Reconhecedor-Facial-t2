# Privacidade e LGPD

Dados biométricos são dados pessoais sensíveis (Lei nº 13.709/2018). Este documento
registra como o projeto os trata. Ele não substitui as obrigações legais do grupo.

## Controles implementados

| Risco | Controle |
|-------|----------|
| Dados no GitHub | Dados ficam fora do repositório; `.gitignore` como segunda barreira; `scripts/privacy_audit.py` verifica índice e histórico |
| Sincronização com a nuvem | A aplicação recusa `FACEREC_HOME` dentro de OneDrive, Dropbox, Google Drive, iCloud ou repositório git |
| Exposição na rede | Servidor apenas em loopback, com validação de `Host` e `Origin` |
| Recursos de terceiros | Sem CDN; `Content-Security-Policy` restritiva |
| Cadastro sem autorização | Confirmação de consentimento obrigatória na interface e na API |
| Excesso de dados | Fotos não são guardadas por padrão; quadros da câmera não são gravados |
| Logs | Apenas evento, identificador parcial e código de erro |
| Retirada do consentimento | Exclusão por pessoa, sem resíduos no arquivo do banco |
| Fim da disciplina | `scripts/purge_all.py --confirmar` |
| Agentes de IA | Recebem código, especificações e dados sintéticos ou públicos, nunca as fotos dos participantes |

## Responsabilidades do grupo

Estes itens são procedimentos humanos que o software não consegue garantir:

- Coletar e guardar os Termos assinados (Anexo A), em local privado e fora do GitHub.
- Usar somente participantes adultos que consentiram.
- Não mover `FACEREC_HOME` para pastas sincronizadas nem enviar seu conteúdo a serviços externos.
- Não publicar checkpoints treinados com dados privados.
- Não usar o sistema para identificar quem não consentiu.
- Usar no vídeo somente quem autorizou essa finalidade no Termo.
- Atender pedidos de retirada excluindo a pessoa do cadastro **e** do `private_dataset`.
- Eliminar os dados ao final da disciplina.

## Checklist antes da entrega (Anexo B)

- [x] Todas as pessoas participantes deram consentimento.
- [x] Os termos de consentimento estão armazenados de forma privada.
- [x] Nenhum termo de consentimento está no GitHub.
- [x] Nenhuma fotografia privada está no GitHub.
- [x] Nenhum embedding privado está no GitHub.
- [x] Nenhum banco biométrico está no GitHub.
- [x] Nenhum checkpoint derivado dos dados privados está no GitHub.
- [x] O `.gitignore` protege os diretórios privados.
- [x] O histórico do Git foi verificado (`python scripts/privacy_audit.py`).
- [ ] As licenças dos datasets públicos foram verificadas.(OBS: os termos do LFW não foram verificados)
- [x] O README informa a origem dos datasets públicos.
- [x] Os resultados publicados não identificam os participantes.
- [x] O vídeo respeita as autorizações concedidas.
- [x] Os dados privados não foram enviados inadvertidamente para serviços externos.
- [x] Existe uma forma de excluir uma pessoa cadastrada.
- [x] O grupo conhece as regras de proteção de dados aplicáveis ao projeto.
