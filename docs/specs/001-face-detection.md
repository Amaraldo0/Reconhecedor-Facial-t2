# SPEC-001: Detecção e alinhamento de faces

## Comportamento

Dada uma imagem RGB, retornar todas as faces detectadas com caixa, confiança, cinco
landmarks e o recorte alinhado de 160×160 pronto para a rede. O alinhamento usa os
olhos para corrigir a rotação antes do recorte, com margem configurável.

A avaliação de qualidade é separada da detecção: o detector relata o que viu, e o
cadastro decide o que aceita.

## Critérios de aceitação

- **CA-001.1** Imagem sem face retorna lista vazia, sem exceção.
- **CA-001.2** Faces abaixo da confiança mínima configurada são descartadas.
- **CA-001.3** Cada face retornada tem recorte 160×160×3.
- **CA-001.4** Arquivo corrompido ou que não é imagem gera `InvalidImageError`.
- **CA-001.5** Imagem maior que o limite configurado de bytes ou pixels gera `InvalidImageError`.
- **CA-001.6** A orientação EXIF é respeitada ao decodificar.
