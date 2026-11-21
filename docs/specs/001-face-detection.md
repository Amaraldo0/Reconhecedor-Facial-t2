# SPEC-001: Detecção e alinhamento de faces

## Comportamento

Dada uma imagem RGB, retornar todas as faces detectadas com caixa, confiança e o recorte
alinhado de 160×160 pronto para a rede. O detector também estima cinco landmarks, e os dos
olhos são usados para corrigir a rotação antes do recorte, com margem configurável.

> **Revisão após a Parte 1:** o experimento E03 comparou o recorte com e sem nivelar os olhos e
> não encontrou diferença (EER de validação 0,0079 nos dois casos). O nivelamento foi mantido
> (`detection.align: true`) por ser o pré-processamento das configurações avaliadas, inclusive a
> escolhida (C01). Pode ser desligado na configuração sem perda medida no LFW.

A avaliação de qualidade é separada da detecção: o detector relata o que viu, e o
cadastro decide o que aceita.

## Critérios de aceitação

- **CA-001.1** Imagem sem face retorna lista vazia, sem exceção.
- **CA-001.2** Faces abaixo da confiança mínima configurada são descartadas.
- **CA-001.3** Cada face retornada tem recorte 160×160×3.
- **CA-001.4** Arquivo corrompido ou que não é imagem gera `InvalidImageError`.
- **CA-001.5** Imagem maior que o limite configurado de bytes ou pixels gera `InvalidImageError`.
- **CA-001.6** A orientação EXIF é respeitada ao decodificar.
