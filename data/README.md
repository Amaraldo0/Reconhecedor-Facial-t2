# Dados

Esta pasta fica vazia de propósito. Nenhum dado de pessoa é versionado.

## LFW (público)

`python scripts/prepare_lfw.py` baixa o dataset da origem, confere a soma SHA-256 e
grava o cache em `FACEREC_HOME/public/lfw`.

## CelebA (público, opcional)

Leia o acordo em <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>. Obtenha as imagens
e a partição na fonte oficial e solicite as identidades conforme o acordo. Coloque
`identity_CelebA.txt` em `FACEREC_HOME/public/celeba`. Depois execute:

```bash
python scripts/prepare_celeba.py --zip caminho/CelebA.zip
python experiments/run_celeba.py
python experiments/summarize_celeba.py
```

## Dataset privado (local)

Crie, **fora do repositório e de pastas sincronizadas**:

```
FACEREC_HOME/private_dataset/<participante>/<fotos>
```

Use somente pessoas adultas que assinaram o Termo de Consentimento (Anexo A do
enunciado). Use identificadores opacos nas pastas (`p01`, `p02`), e não nomes.
Guarde os termos separadamente, em local privado, nunca no GitHub.

Procure variar iluminação, pose, distância, expressão, fundo, uso de óculos e momento
de captura, coletando apenas o necessário.

Depois execute `python experiments/evaluate_private.py`. A saída é somente agregada.
