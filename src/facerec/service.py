"""Casos de uso da aplicação: cadastrar, reconhecer e excluir."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from .config import AppConfig
from .errors import (
    ConsentRequiredError,
    DuplicateFaceError,
    FaceRecError,
    InadequateImageError,
    InconsistentIdentityError,
    InvalidImageError,
    InvalidNameError,
    MultipleFacesError,
    NoFaceError,
)
from .imaging import decode_image
from .matching import best_match, is_recognized
from .quality import inadequacy_reason
from .storage import FaceStore, Person, name_key
from .vision import Detector, Embedder

# Por política (CA-002.10), logs nunca recebem nomes, imagens, embeddings ou similaridades.
log = logging.getLogger("facerec")


@dataclass(frozen=True)
class FaceResult:
    box: tuple[float, float, float, float]
    recognized: bool
    person_id: str | None
    name: str | None
    similarity: float | None


class FaceService:
    def __init__(self, cfg: AppConfig, detector: Detector, embedder: Embedder, store: FaceStore):
        self.cfg = cfg
        self.detector = detector
        self.embedder = embedder
        self.store = store
        self._inference_lock = threading.Lock()
        self._gallery_version = -1
        self._gallery = np.empty((0, 512), dtype=np.float32)
        self._owners: list[str] = []
        self.skipped_embeddings = store.count_other_models(embedder.model_id)
        if self.skipped_embeddings:
            log.warning("gallery_model_mismatch count=%d", self.skipped_embeddings)

    def _current_gallery(self) -> tuple[np.ndarray, list[str]]:
        if self._gallery_version != self.store.version:
            self._gallery_version = self.store.version
            self._gallery, self._owners = self.store.load_gallery(self.embedder.model_id)
        return self._gallery, self._owners

    def _decode(self, data: bytes) -> np.ndarray:
        return decode_image(data, self.cfg.limits.max_upload_bytes, self.cfg.limits.max_pixels)

    def enroll(self, name: str, consent: bool, images: list[bytes]) -> Person:
        if not consent:
            raise ConsentRequiredError()
        clean_name = " ".join((name or "").split())
        if not clean_name or len(clean_name) > 100 or not name_key(clean_name):
            raise InvalidNameError()
        if not images:
            raise InvalidImageError("Envie ao menos uma foto.")
        if len(images) > self.cfg.enrollment.max_images:
            raise InvalidImageError(f"Envie no máximo {self.cfg.enrollment.max_images} fotos.")

        crops = []
        for index, data in enumerate(images):
            try:
                crops.append(self._single_face_crop(data))
            except FaceRecError as error:
                error.image_index = index
                log.info("enroll_rejected code=%s image=%d", error.code, index)
                raise

        with self._inference_lock:
            embeddings = self.embedder.embed(np.stack(crops))
        self._assert_same_identity(embeddings)
        self._assert_not_enrolled(embeddings)

        photos = list(images) if self.cfg.storage.store_photos else None
        person = self.store.add_person(clean_name, embeddings, self.embedder.model_id, photos)
        log.info("enroll_ok person=%s images=%d", person.id[:8], len(images))
        return person

    def _single_face_crop(self, data: bytes) -> np.ndarray:
        image = self._decode(data)
        with self._inference_lock:
            faces = self.detector.detect(image)
        if not faces:
            raise NoFaceError()
        if len(faces) > 1:
            raise MultipleFacesError()
        if reason := inadequacy_reason(faces[0], self.cfg.quality):
            raise InadequateImageError(reason)
        return faces[0].crop

    def _assert_same_identity(self, embeddings: np.ndarray) -> None:
        if len(embeddings) < 2:
            return
        similarities = embeddings @ embeddings.T
        if similarities.min() < self.cfg.enrollment.consistency_threshold:
            raise InconsistentIdentityError()

    def _assert_not_enrolled(self, embeddings: np.ndarray) -> None:
        gallery, owners = self._current_gallery()
        for embedding in embeddings:
            match = best_match(embedding, gallery, owners, self.cfg.recognition.aggregation)
            if match and is_recognized(match.similarity, self.cfg.recognition.threshold):
                raise DuplicateFaceError()

    def recognize(self, data: bytes) -> list[FaceResult]:
        """Classifica cada face da imagem. Nada da consulta é armazenado (CA-003.5)."""
        image = self._decode(data)
        with self._inference_lock:
            faces = self.detector.detect(image)
            if not faces:
                return []
            embeddings = self.embedder.embed(np.stack([face.crop for face in faces]))
        gallery, owners = self._current_gallery()
        results = []
        for face, embedding in zip(faces, embeddings):
            match = best_match(embedding, gallery, owners, self.cfg.recognition.aggregation)
            if match and is_recognized(match.similarity, self.cfg.recognition.threshold):
                results.append(
                    FaceResult(face.box, True, match.person_id, self.store.name_of(match.person_id), match.similarity)
                )
            else:
                results.append(FaceResult(face.box, False, None, None, None))
        return results

    def list_persons(self) -> list[Person]:
        return self.store.list_persons()

    def delete(self, person_id: str) -> None:
        self.store.delete_person(person_id)
        log.info("delete_ok person=%s", person_id[:8])
