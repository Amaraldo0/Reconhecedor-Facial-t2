# SPEC-004: Pessoa desconhecida

## Regra de decisão

```
similaridade >= threshold  → pessoa reconhecida
similaridade <  threshold  → pessoa desconhecida
```

O valor do threshold é definido no conjunto de **validação** (ADR-003), nunca no de
teste, e fica na configuração.

Para uma face desconhecida, a resposta não revela o nome do candidato mais próximo.

## Critérios de aceitação

- **CA-004.1** Similaridade igual ao threshold é reconhecida.
- **CA-004.2** Similaridade abaixo do threshold é desconhecida.
- **CA-004.3** Resposta "desconhecida" não contém nome nem identificador de candidato.
- **CA-004.4** O threshold vem da configuração, sem valor fixo no código de decisão.
