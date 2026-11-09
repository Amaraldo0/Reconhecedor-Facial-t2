# SPEC-000: Visão geral e requisitos

## Problema

Identificar, em uma imagem ou no vídeo de uma câmera, pessoas que **consentiram** e
foram cadastradas, respondendo "pessoa desconhecida" para qualquer outra face.
Trata-se de um problema de **identificação em conjunto aberto**: não há garantia de
que a face observada pertença ao cadastro.

## Pipeline

```
Imagem → Detecção → Alinhamento → Rede neural → Embedding → Comparação → Threshold → Decisão
```

## Requisitos

| ID | Requisito | Spec |
|----|-----------|------|
| RF-1 | Detectar e alinhar faces em uma imagem | 001 |
| RF-2 | Cadastrar uma pessoa, tratando nenhuma face, múltiplas faces, imagem inadequada e duplicidade | 002 |
| RF-3 | Reconhecer pessoas em imagem ou câmera | 003 |
| RF-4 | Responder "desconhecido" abaixo do limiar | 004 |
| RF-5 | Interface com câmera | 005 |
| RF-6 | Excluir uma pessoa e todos os dados associados | 006 |
| RF-7 | Determinar experimentalmente a melhor configuração | 007 |
| RNF-1 | Dados privados nunca entram no repositório nem saem da máquina | 006, `docs/privacy/LGPD.md` |
| RNF-2 | Experimentos reproduzíveis, sem vazamento entre treino, validação e teste | 007 |
| RNF-3 | Configuração externa, logs sem dados pessoais, erros tratados | 002, 003 |
| RNF-4 | Um clone novo começa com "Nenhuma pessoa cadastrada" | 005 |

## Fora de escopo

Vigilância, monitoramento ou identificação de quem não consentiu (proibidos pelo
enunciado), prova de vida (anti-spoofing) e acesso pela rede.

## Fluxo de desenvolvimento

Problema → Requisitos → Especificação → Critérios de aceitação → Planejamento →
Implementação → Testes → Revisão. Cada critério de aceitação (`CA-xxx`) é citado no
teste que o verifica. Decisões técnicas ficam em `docs/decisions/`.
