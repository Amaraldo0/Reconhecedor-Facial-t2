"""Critérios da SPEC-005 e contratos HTTP."""
from __future__ import annotations

from pathlib import Path

from conftest import make_image
from facerec.app import server


def enroll(client, name="Ana", identity=1, consent="true"):
    return client.post(
        "/api/persons", data={"name": name, "consent": consent},
        files=[("files", ("a.png", make_image(identity), "image/png"))],
    )


def test_ca_005_1_clone_novo_sem_pessoas(client):
    assert client.get("/api/status").json()["persons"] == 0
    assert client.get("/api/persons").json() == []
    assert "Nenhuma pessoa cadastrada." in client.get("/").text


def test_ca_005_3_host_externo_recusado(client):
    assert client.get("/api/status", headers={"host": "atacante.example"}).status_code == 403


def test_ca_005_3_origin_externo_recusado(client):
    response = client.delete("/api/persons/x", headers={"origin": "https://atacante.example"})
    assert response.status_code == 403


def test_ca_005_3_origin_local_aceito(client):
    assert client.get("/api/status", headers={"origin": "http://testserver"}).status_code == 200


def test_cadastro_reconhecimento_exclusao(client):
    created = enroll(client)
    assert created.status_code == 201
    faces = client.post("/api/recognize", files={"file": ("q.png", make_image(1), "image/png")}).json()["faces"]
    assert faces[0]["recognized"] and faces[0]["name"] == "Ana"
    assert client.delete(f"/api/persons/{created.json()['id']}").status_code == 204
    faces = client.post("/api/recognize", files={"file": ("q.png", make_image(1), "image/png")}).json()["faces"]
    assert faces[0]["recognized"] is False


def test_ca_004_3_resposta_http_de_desconhecido(client):
    enroll(client)
    [face] = client.post("/api/recognize", files={"file": ("q.png", make_image(2), "image/png")}).json()["faces"]
    assert face["name"] is None and face["person_id"] is None and face["similarity"] is None


def test_ca_005_5_erro_em_portugues_com_indice(client):
    response = client.post(
        "/api/persons", data={"name": "Ana", "consent": "true"},
        files=[("files", ("a.png", make_image(faces=0), "image/png"))],
    )
    body = response.json()
    assert response.status_code == 422 and body["code"] == "nenhuma_face" and body["image_index"] == 0
    assert "Nenhuma face" in body["message"]


def test_consentimento_ausente(client):
    assert enroll(client, consent="false").json()["code"] == "consentimento_obrigatorio"


def test_nome_duplicado_409(client):
    enroll(client, identity=1)
    assert enroll(client, identity=2).status_code == 409


def test_ca_006_7_excluir_inexistente_404(client):
    assert client.delete("/api/persons/nao-existe").status_code == 404


def test_ca_005_7_pagina_sem_recursos_externos(client):
    static = Path(server.STATIC)
    for file in ("index.html", "app.js", "style.css"):
        content = (static / file).read_text(encoding="utf-8")
        assert "http://" not in content and "https://" not in content
    assert "default-src 'none'" in client.get("/").headers["content-security-policy"]
