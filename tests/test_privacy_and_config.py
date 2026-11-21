"""Critérios de privacidade (SPEC-006) e decodificação (SPEC-001)."""
from __future__ import annotations

import io
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from facerec.config import REPO_ROOT, assert_safe_data_dir, load_config
from facerec.errors import UnsafeDataDirError
from facerec.imaging import align_and_crop, decode_image

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import prepare_celeba  # noqa: E402
import privacy_audit  # noqa: E402
import purge_all  # noqa: E402


def test_ca_006_5_recusa_pasta_sincronizada(tmp_path):
    with pytest.raises(UnsafeDataDirError):
        assert_safe_data_dir(tmp_path / "OneDrive" / "facerec")


def test_ca_006_5_recusa_repositorio_git(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(UnsafeDataDirError):
        assert_safe_data_dir(tmp_path / "dados")


def test_ca_006_5_load_config_recusa(tmp_path):
    with pytest.raises(UnsafeDataDirError):
        load_config(home=tmp_path / "Dropbox" / "x")


def test_diretorio_local_aceito(tmp_path):
    assert_safe_data_dir(tmp_path / "dados")


@pytest.mark.parametrize("path", [
    "data/private/a.jpg", "data/familia/a.txt", "data/amigos/x", "data/teste_local/x", "faces/a.txt",
    "embeddings/a.txt", "database/x.txt", "checkpoints/private/m.bin", "consent/termo.pdf", "x.db", "x.sqlite",
    "model.pt", "model.pth", "model.ckpt", "model.onnx", "weights.bin", "adapter.safetensors", "e.npy", "foto.jpg",
])
def test_ca_006_6_gitignore_protege(path):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT)
    assert result.returncode == 0, path


def test_auditoria_detecta_arquivos_proibidos():
    assert privacy_audit.violations({"src/a.py", "docs/specs/x.md"}) == []
    assert privacy_audit.violations({"faces/ana.txt", "m.pt", "foto.png", "data/x.csv"}) == [
        "data/x.csv", "faces/ana.txt", "foto.png", "m.pt",
    ]
    assert privacy_audit.violations({"experiments/results/figures/roc.png"}) == []


def test_purge_remove_privados_e_mantem_publicos(tmp_path):
    for relative in ("gallery/a", "private_dataset/p/f", "public/lfw/x"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("x")
    purge_all.purge(tmp_path)
    assert not (tmp_path / "gallery").exists() and not (tmp_path / "private_dataset").exists()
    assert (tmp_path / "public/lfw/x").exists()


def test_ca_008_1_md5_divergente_interrompe_preparacao():
    import hashlib

    prepare_celeba.verify_md5(b"conteudo", hashlib.md5(b"conteudo").hexdigest())
    with pytest.raises(SystemExit):
        prepare_celeba.verify_md5(b"arquivo adulterado")


def test_ca_001_6_respeita_exif():
    image = Image.new("RGB", (40, 20))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotação de 90 graus
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    assert decode_image(buffer.getvalue(), 10**6, 10**6).shape[:2] == (40, 20)


def test_ca_001_3_recorte_no_tamanho_configurado():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    crop = align_and_crop(image, np.array([50, 50, 200, 220]), np.array([90, 100]), np.array([160, 110]), 160, 14)
    assert crop.shape == (160, 160, 3)


def test_alinhamento_nivela_os_olhos():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    left, right = np.array([100.0, 100.0]), np.array([200.0, 160.0])
    image[97:104, 97:104] = 255
    image[157:164, 197:204] = 255
    crop = align_and_crop(image, np.array([60, 50, 240, 230]), left, right, 160, 0)
    rows = np.flatnonzero(crop[:, :, 0].max(axis=1) > 128)
    assert rows.max() - rows.min() < 12


def test_ca_005_2_config_padrao_loopback(tmp_path):
    assert load_config(home=tmp_path).server.host == "127.0.0.1"
