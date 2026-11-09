"""Erros de domínio, com código estável e mensagem em português para o usuário."""
from __future__ import annotations


class FaceRecError(Exception):
    code = "erro"
    http_status = 400
    default_message = "Não foi possível concluir a operação."

    def __init__(self, message: str | None = None, image_index: int | None = None):
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        self.image_index = image_index


class ConsentRequiredError(FaceRecError):
    code = "consentimento_obrigatorio"
    default_message = "Confirme que a pessoa assinou o Termo de Consentimento antes do cadastro."


class InvalidNameError(FaceRecError):
    code = "nome_invalido"
    default_message = "Informe um nome com até 100 caracteres."


class InvalidImageError(FaceRecError):
    code = "imagem_invalida"
    default_message = "O arquivo não é uma imagem válida. Envie um JPEG ou PNG."


class NoFaceError(FaceRecError):
    code = "nenhuma_face"
    http_status = 422
    default_message = "Nenhuma face foi encontrada. Use uma foto com o rosto visível e de frente."


class MultipleFacesError(FaceRecError):
    code = "multiplas_faces"
    http_status = 422
    default_message = "Há mais de uma face na imagem. Use uma foto só com a pessoa a cadastrar."


class InadequateImageError(FaceRecError):
    code = "imagem_inadequada"
    http_status = 422
    default_message = "A qualidade da imagem é insuficiente."


class InconsistentIdentityError(FaceRecError):
    code = "identidades_divergentes"
    http_status = 422
    default_message = "As fotos enviadas parecem ser de pessoas diferentes."


class DuplicatePersonError(FaceRecError):
    code = "nome_duplicado"
    http_status = 409
    default_message = "Já existe uma pessoa cadastrada com esse nome."


class DuplicateFaceError(FaceRecError):
    code = "face_duplicada"
    http_status = 409
    default_message = "Esta face já está cadastrada sob outro nome."


class PersonNotFoundError(FaceRecError):
    code = "pessoa_nao_encontrada"
    http_status = 404
    default_message = "Pessoa não encontrada."


class UnsafeDataDirError(RuntimeError):
    """O diretório de dados privados está em local que poderia expô-los."""
