"""Critérios das SPECs 001 a 004 e 006, na camada de serviço."""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pytest

from conftest import BLURRY, DARK, SMALL, FakeDetector, FakeEmbedder, make_image
from facerec import errors
from facerec.config import LimitsConfig, RecognitionConfig, StorageConfig
from facerec.matching import best_match, is_recognized
from facerec.service import FaceService
from facerec.storage import FaceStore


def test_ca_002_1_sem_consentimento_nada_e_gravado(service):
    with pytest.raises(errors.ConsentRequiredError):
        service.enroll("Ana", False, [make_image()])
    assert service.list_persons() == []


def test_ca_002_2_sem_face(service):
    with pytest.raises(errors.NoFaceError):
        service.enroll("Ana", True, [make_image(faces=0)])


def test_ca_002_3_multiplas_faces(service):
    with pytest.raises(errors.MultipleFacesError):
        service.enroll("Ana", True, [make_image(faces=2)])


@pytest.mark.parametrize("problem,word", [(BLURRY, "borrada"), (DARK, "escura"), (SMALL, "pequeno")])
def test_ca_002_4_imagem_inadequada_informa_motivo(service, problem, word):
    with pytest.raises(errors.InadequateImageError, match=word):
        service.enroll("Ana", True, [make_image(problem=problem)])


@pytest.mark.parametrize("variant", ["joao silva", "JOÃO   SILVA", "João Silva"])
def test_ca_002_5_nome_duplicado(service, variant):
    # Usa outra face de propósito: com a mesma face, a recusa viria de CA-002.6
    # (face duplicada) antes de chegar à verificação do nome.
    service.enroll("João Silva", True, [make_image(1)])
    with pytest.raises(errors.DuplicatePersonError):
        service.enroll(variant, True, [make_image(2)])


def test_ca_002_6_face_ja_cadastrada_com_outro_nome(service):
    service.enroll("Ana", True, [make_image(1)])
    with pytest.raises(errors.DuplicateFaceError):
        service.enroll("Beatriz", True, [make_image(1)])


def test_ca_002_7_falha_em_uma_imagem_nao_grava_nada(service):
    with pytest.raises(errors.NoFaceError) as caught:
        service.enroll("Ana", True, [make_image(1), make_image(1, faces=0)])
    assert caught.value.image_index == 1
    assert service.list_persons() == []


def test_ca_002_8_nao_grava_fotos_por_padrao(service, config):
    service.enroll("Ana", True, [make_image(1)])
    assert not config.photos_dir.exists() or not list(config.photos_dir.rglob("*"))


def test_ca_002_9_identidades_divergentes(service):
    with pytest.raises(errors.InconsistentIdentityError):
        service.enroll("Ana", True, [make_image(1), make_image(2)])


def test_ca_002_10_logs_sem_nome(service, caplog):
    with caplog.at_level(logging.INFO, logger="facerec"):
        person = service.enroll("Nome Sigiloso", True, [make_image(1)])
        service.recognize(make_image(1))
        service.delete(person.id)
    assert caplog.text and "Sigiloso" not in caplog.text


def test_ca_003_1_reconhece_com_nome_e_similaridade(service):
    service.enroll("Ana", True, [make_image(1)])
    [face] = service.recognize(make_image(1))
    assert face.recognized and face.name == "Ana" and face.similarity == pytest.approx(1.0)


def test_ca_003_2_uma_decisao_por_face(service):
    service.enroll("Ana", True, [make_image(1)])
    assert len(service.recognize(make_image(1, faces=3))) == 3


def test_ca_003_3_sem_faces_nao_e_erro(service):
    assert service.recognize(make_image(faces=0)) == []


def test_ca_003_4_embeddings_de_outro_modelo_nao_sao_comparados(config, store):
    other = FakeEmbedder()
    other.model_id = "outro"
    FaceService(config, FakeDetector(), other, store).enroll("Ana", True, [make_image(1)])
    current = FaceService(config, FakeDetector(), FakeEmbedder(), store)
    assert current.skipped_embeddings == 1
    assert current.recognize(make_image(1))[0].recognized is False


def test_ca_003_5_reconhecimento_nao_grava(service, store, config):
    service.enroll("Ana", True, [make_image(1)])
    before = store.version, store.list_persons()
    service.recognize(make_image(1))
    assert (store.version, store.list_persons()) == before


def test_ca_003_6_mudanca_na_galeria_reflete_na_consulta(service):
    assert service.recognize(make_image(1))[0].recognized is False
    service.enroll("Ana", True, [make_image(1)])
    assert service.recognize(make_image(1))[0].recognized is True


def test_galeria_vazia_retorna_desconhecido(service):
    [face] = service.recognize(make_image(1))
    assert not face.recognized and face.name is None


def test_ca_004_1_igual_ao_threshold_e_reconhecido():
    assert is_recognized(0.5, 0.5)


def test_ca_004_2_abaixo_do_threshold_e_desconhecido():
    assert not is_recognized(0.4999, 0.5)


def test_ca_004_1_servico_aceita_exatamente_no_threshold(config, store):
    at_limit = replace(config, recognition=RecognitionConfig(threshold=1.0))
    service = FaceService(at_limit, FakeDetector(), FakeEmbedder(), store)
    service.enroll("Ana", True, [make_image(1)])
    assert service.recognize(make_image(1))[0].recognized is True


def test_ca_004_2_servico_rejeita_logo_abaixo_do_threshold(config, store):
    above = replace(config, recognition=RecognitionConfig(threshold=float(np.nextafter(np.float32(1.0), np.float32(2.0)))))
    service = FaceService(above, FakeDetector(), FakeEmbedder(), store)
    service.enroll("Ana", True, [make_image(1)])
    assert service.recognize(make_image(1))[0].recognized is False


def test_ca_004_3_desconhecido_nao_revela_candidato(service):
    service.enroll("Ana", True, [make_image(1)])
    [face] = service.recognize(make_image(2))
    assert (face.recognized, face.person_id, face.name, face.similarity) == (False, None, None, None)


def test_ca_004_4_threshold_vem_da_configuracao(config, store):
    strict = replace(config, recognition=RecognitionConfig(threshold=1.01))
    service = FaceService(strict, FakeDetector(), FakeEmbedder(), store)
    service.enroll("Ana", True, [make_image(1)])
    assert service.recognize(make_image(1))[0].recognized is False


def test_ca_001_4_arquivo_invalido(service):
    with pytest.raises(errors.InvalidImageError):
        service.recognize(b"isto nao e uma imagem")


def test_ca_001_5_arquivo_acima_do_limite(config, store):
    limited = replace(config, limits=LimitsConfig(max_upload_bytes=10, max_pixels=1000))
    service = FaceService(limited, FakeDetector(), FakeEmbedder(), store)
    with pytest.raises(errors.InvalidImageError):
        service.recognize(make_image())


def test_ca_006_1_e_3_exclusao_remove_linhas_e_reconhecimento(service, store):
    person = service.enroll("Ana", True, [make_image(1)])
    service.delete(person.id)
    rows = [store._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("persons", "embeddings")]
    assert rows == [0, 0]
    assert service.recognize(make_image(1))[0].recognized is False


def test_ca_006_2_exclusao_remove_fotos(config, store):
    storing = replace(config, storage=StorageConfig(store_photos=True))
    service = FaceService(storing, FakeDetector(), FakeEmbedder(), store)
    person = service.enroll("Ana", True, [make_image(1)])
    assert list((config.photos_dir / person.id).iterdir())
    service.delete(person.id)
    assert not (config.photos_dir / person.id).exists()


def test_ca_006_4_sem_residuos_no_arquivo_do_banco(service, config):
    person = service.enroll("NomeResidual", True, [make_image(7)])
    vector = np.zeros(512, dtype=np.float32)
    vector[7] = 1.0
    assert vector.tobytes() in config.db_path.read_bytes()
    service.delete(person.id)
    content = config.db_path.read_bytes()
    assert b"NomeResidual" not in content and vector.tobytes() not in content


def test_ca_006_7_excluir_inexistente(service):
    with pytest.raises(errors.PersonNotFoundError):
        service.delete("nao-existe")


def test_prototype_agrega_pela_media():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    match = best_match(np.array([1.0, 0.0]), gallery, ["a", "a", "b"], "prototype")
    assert match.person_id == "b" and match.similarity == pytest.approx(0.8)


def test_persistencia_entre_execucoes(config, store):
    service = FaceService(config, FakeDetector(), FakeEmbedder(), store)
    service.enroll("Ana", True, [make_image(1)])
    reopened = FaceStore(config.db_path, config.photos_dir)
    try:
        assert FaceService(config, FakeDetector(), FakeEmbedder(), reopened).recognize(make_image(1))[0].name == "Ana"
    finally:
        reopened.close()
