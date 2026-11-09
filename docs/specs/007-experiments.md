# SPEC-007: Protocolo experimental (Parte 1)

## Pergunta

Qual configuração de modelo, treinamento e fine-tuning apresenta o melhor desempenho?

## Dados

- **Público**: LFW, alinhado com o mesmo detector da aplicação. Termos e fonte no README.
- **Privado**: fotos dos integrantes e participantes com consentimento, processadas
  somente na máquina do grupo por `experiments/evaluate_private.py`, com saída
  apenas agregada.

## Separação e vazamento

A divisão é **por identidade**, com semente fixa: 60% treino, 20% validação, 20% teste.
Nenhuma pessoa aparece em mais de um conjunto.

Antes de treinar, um hash perceptual procura imagens iguais ou quase iguais entre
conjuntos. As ocorrências são removidas do conjunto de treino ou validação e o total é
relatado. Aumento de dados só é aplicado depois da divisão e só no treino.

O conjunto de teste não é usado para treino, parada antecipada, escolha de
hiperparâmetros, threshold ou modelo. Essas escolhas usam a validação.

## Avaliação

- **Verificação**: pares balanceados, com limite de imagens por identidade. ROC-AUC,
  EER, accuracy, precision, recall, F1, FAR e FRR no threshold escolhido na validação,
  e distribuições de similaridade (mesma pessoa e pessoas diferentes).
- **Identificação em conjunto aberto**: identidades conhecidas e desconhecidas. Taxa
  de acerto entre conhecidos e taxa de falsa aceitação entre desconhecidos.
- **Robustez**: as mesmas métricas no teste degradado (baixa resolução, desfoque,
  variação de brilho).
- **Incerteza**: intervalo de confiança de 95% por bootstrap sobre identidades.

## Registro

Cada experimento grava hipótese, configuração, dataset, modelo, estratégia,
hiperparâmetros, métricas, tempo, versões das bibliotecas, semente e hardware em
`experiments/results/<id>.json`. Checkpoints ficam em `FACEREC_HOME`, nunca no repo.

## Critérios de aceitação

- **CA-007.1** Treino, validação e teste têm identidades disjuntas.
- **CA-007.2** A mesma semente produz a mesma divisão.
- **CA-007.3** O threshold é calculado sem acesso às pontuações de teste.
- **CA-007.4** Hash perceptual não encontra par entre conjuntos depois da remoção.
- **CA-007.5** As métricas conferem com casos calculados à mão.
- **CA-007.6** `evaluate_private.py` não imprime nem grava resultado individualizado.
