# Rastreabilidade dos critérios de aceitação

Cada critério das SPECs aponta para o teste que o verifica. Onde um teste automatizado
seria artificial, como comportamento visual no navegador, consta a verificação manual
realizada. Testes marcados como *slow* exigem os pesos reais (`pytest -m slow`).

| Critério | Verificação |
|---|---|
| CA-001.1 | `test_real_model.py::test_ca_001_1_imagem_sem_face_retorna_lista_vazia` (slow) |
| CA-001.2 | `test_real_model.py::test_ca_001_2_descarta_faces_abaixo_da_confianca` (slow) |
| CA-001.3 | `test_privacy_and_config.py::test_ca_001_3_recorte_no_tamanho_configurado` |
| CA-001.4 | `test_service.py::test_ca_001_4_arquivo_invalido` |
| CA-001.5 | `test_service.py::test_ca_001_5_arquivo_acima_do_limite` |
| CA-001.6 | `test_privacy_and_config.py::test_ca_001_6_respeita_exif` |
| CA-002.1 | `test_service.py::test_ca_002_1_sem_consentimento_nada_e_gravado`; `test_api.py::test_consentimento_ausente` |
| CA-002.2 | `test_service.py::test_ca_002_2_sem_face` |
| CA-002.3 | `test_service.py::test_ca_002_3_multiplas_faces` |
| CA-002.4 | `test_service.py::test_ca_002_4_imagem_inadequada_informa_motivo` |
| CA-002.5 | `test_service.py::test_ca_002_5_nome_duplicado` |
| CA-002.6 | `test_service.py::test_ca_002_6_face_ja_cadastrada_com_outro_nome` |
| CA-002.7 | `test_service.py::test_ca_002_7_falha_em_uma_imagem_nao_grava_nada` |
| CA-002.8 | `test_service.py::test_ca_002_8_nao_grava_fotos_por_padrao` |
| CA-002.9 | `test_service.py::test_ca_002_9_identidades_divergentes` |
| CA-002.10 | `test_service.py::test_ca_002_10_logs_sem_nome` |
| CA-003.1 | `test_service.py::test_ca_003_1_reconhece_com_nome_e_similaridade` |
| CA-003.2 | `test_service.py::test_ca_003_2_uma_decisao_por_face` |
| CA-003.3 | `test_service.py::test_ca_003_3_sem_faces_nao_e_erro` |
| CA-003.4 | `test_service.py::test_ca_003_4_embeddings_de_outro_modelo_nao_sao_comparados` |
| CA-003.5 | `test_service.py::test_ca_003_5_reconhecimento_nao_grava` |
| CA-003.6 | `test_service.py::test_ca_003_6_mudanca_na_galeria_reflete_na_consulta` |
| CA-004.1 | `test_service.py::test_ca_004_1_igual_ao_threshold_e_reconhecido`; `test_ca_004_1_servico_aceita_exatamente_no_threshold` |
| CA-004.2 | `test_service.py::test_ca_004_2_abaixo_do_threshold_e_desconhecido`; `test_ca_004_2_servico_rejeita_logo_abaixo_do_threshold` |
| CA-004.3 | `test_service.py::test_ca_004_3_desconhecido_nao_revela_candidato`; `test_api.py::test_ca_004_3_resposta_http_de_desconhecido` |
| CA-004.4 | `test_service.py::test_ca_004_4_threshold_vem_da_configuracao` |
| CA-005.1 | `test_api.py::test_ca_005_1_clone_novo_sem_pessoas` |
| CA-005.2 | `test_privacy_and_config.py::test_ca_005_2_config_padrao_loopback`; a recusa de host fora do loopback está em `app/__main__.py` |
| CA-005.3 | `test_api.py::test_ca_005_3_host_externo_recusado`, `test_ca_005_3_origin_externo_recusado`, `test_ca_005_3_origin_local_aceito` |
| CA-005.4 | **Manual**, no navegador (09/11/2026): caixa verde com nome para a pessoa cadastrada e caixa vermelha com "Pessoa desconhecida" na mesma imagem |
| CA-005.5 | `test_api.py::test_ca_005_5_erro_em_portugues_com_indice`; mensagem exibida na interface conferida manualmente |
| CA-005.6 | **Manual**, no navegador (09/11/2026): com a câmera bloqueada, o aviso aparece e o envio de imagem continua funcionando |
| CA-005.7 | `test_api.py::test_ca_005_7_pagina_sem_recursos_externos` |
| CA-006.1 | `test_service.py::test_ca_006_1_e_3_exclusao_remove_linhas_e_reconhecimento` |
| CA-006.2 | `test_service.py::test_ca_006_2_exclusao_remove_fotos` |
| CA-006.3 | `test_service.py::test_ca_006_1_e_3_exclusao_remove_linhas_e_reconhecimento`; `test_api.py::test_cadastro_reconhecimento_exclusao` |
| CA-006.4 | `test_service.py::test_ca_006_4_sem_residuos_no_arquivo_do_banco` |
| CA-006.5 | `test_privacy_and_config.py::test_ca_006_5_*` (pasta sincronizada, repositório git, `load_config`) |
| CA-006.6 | `test_privacy_and_config.py::test_ca_006_6_gitignore_protege` |
| CA-006.7 | `test_service.py::test_ca_006_7_excluir_inexistente`; `test_api.py::test_ca_006_7_excluir_inexistente_404` |
| CA-007.1 | `test_experiments.py::test_ca_007_1_identidades_disjuntas` |
| CA-007.2 | `test_experiments.py::test_ca_007_2_mesma_semente_mesma_divisao` |
| CA-007.3 | `test_experiments.py::test_ca_007_3_threshold_nao_depende_do_teste` |
| CA-007.4 | `test_experiments.py::test_ca_007_4_remove_duplicatas_entre_conjuntos` |
| CA-007.5 | `test_experiments.py::test_ca_007_5_metricas_calculadas_a_mao`, `test_ca_007_5_eer_separacao_perfeita` |
| CA-007.6 | `test_experiments.py::test_ca_007_6_avaliacao_privada_so_tem_agregados`, `test_ca_007_6_recusa_grupo_pequeno_demais` |
| CA-008.1 | `test_privacy_and_config.py::test_ca_008_1_md5_divergente_interrompe_preparacao` |
| CA-008.2 | `test_experiments.py::test_ca_008_2_identidade_nao_pode_cruzar_particoes` |
| CA-008.3 | `test_experiments.py::test_ca_008_3_limite_por_identidade` |
| CA-008.4 | `test_experiments.py::test_ca_008_4_remove_so_identidades_parecidas` |
| CA-008.5 | **Inspeção**: os scripts gravam recortes e checkpoints só em caminhos derivados de `FACEREC_HOME`, protegido por `assert_safe_data_dir`; uma segunda barreira é o `.gitignore` (`*.npy`, `*.pt`, `checkpoints/`), verificado em CA-006.6 e por `scripts/privacy_audit.py` |
