"""API HTTP local e entrega da interface (SPEC-005)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import AppConfig
from ..errors import FaceRecError
from ..service import FaceService

STATIC = Path(__file__).parent / "static"
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
    "media-src 'self' blob:; connect-src 'self'; base-uri 'none'; form-action 'none'"
)
log = logging.getLogger("facerec")


def allowed_authorities(cfg: AppConfig) -> set[str]:
    port = cfg.server.port
    return {f"127.0.0.1:{port}", f"localhost:{port}"}


def create_app(service: FaceService, cfg: AppConfig, extra_hosts: set[str] | None = None) -> FastAPI:
    app = FastAPI(title="Reconhecimento Facial UFMS", docs_url=None, redoc_url=None, openapi_url=None)
    hosts = allowed_authorities(cfg) | (extra_hosts or set())
    origins = {f"http://{host}" for host in hosts}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        """Bloqueia DNS rebinding e requisições disparadas por outros sites (CA-005.3)."""
        origin = request.headers.get("origin")
        if request.headers.get("host") not in hosts or (origin is not None and origin not in origins):
            return JSONResponse({"code": "acesso_negado", "message": "Acesso permitido apenas localmente."}, 403)
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(FaceRecError)
    async def domain_error(_: Request, error: FaceRecError):
        body = {"code": error.code, "message": error.message}
        if error.image_index is not None:
            body["image_index"] = error.image_index
        return JSONResponse(body, error.http_status)

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, error: Exception):
        log.error("unexpected_error type=%s", type(error).__name__)
        return JSONResponse({"code": "erro_interno", "message": "Erro interno. Consulte os logs."}, 500)

    def read_upload(upload: UploadFile) -> bytes:
        return upload.file.read(cfg.limits.max_upload_bytes + 1)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return {
            "persons": len(service.list_persons()),
            "model_id": service.embedder.model_id,
            "skipped_embeddings": service.skipped_embeddings,
            "max_images": cfg.enrollment.max_images,
        }

    @app.get("/api/persons")
    def persons():
        return [
            {"id": p.id, "name": p.name, "created_at": p.created_at, "num_embeddings": p.num_embeddings}
            for p in service.list_persons()
        ]

    @app.post("/api/persons", status_code=201)
    def enroll(
        name: Annotated[str, Form()],
        files: Annotated[list[UploadFile], File()],
        consent: Annotated[bool, Form()] = False,
    ):
        person = service.enroll(name, consent, [read_upload(file) for file in files])
        return {"id": person.id, "name": person.name, "num_embeddings": person.num_embeddings}

    @app.delete("/api/persons/{person_id}", status_code=204)
    def delete(person_id: str):
        service.delete(person_id)

    @app.post("/api/recognize")
    def recognize(file: Annotated[UploadFile, File()]):
        return {
            "faces": [
                {
                    "box": list(result.box),
                    "recognized": result.recognized,
                    "person_id": result.person_id,
                    "name": result.name,
                    "similarity": None if result.similarity is None else round(result.similarity, 3),
                }
                for result in service.recognize(read_upload(file))
            ]
        }

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
