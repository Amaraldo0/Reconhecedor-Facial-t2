"use strict";

const $ = (id) => document.getElementById(id);
const MAX_FRAME_WIDTH = 960;
const COLORS = { ok: "#22a653", bad: "#e0483e" };
const CAMERA_ERROR = "Não foi possível acessar a câmera. Verifique a permissão do navegador ou envie uma imagem.";

let maxImages = 10;
let cameraStream = null;
let enrollStream = null;
let pendingPhotos = [];

function showMessage(element, text, ok = false) {
  element.textContent = text;
  element.classList.toggle("ok", ok);
  element.hidden = false;
}

function hideMessage(element) {
  element.hidden = true;
  element.textContent = "";
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("O servidor local não respondeu. Verifique se a aplicação está em execução.");
  }
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const prefix = Number.isInteger(body.image_index) ? `Foto ${body.image_index + 1}: ` : "";
    throw new Error(prefix + (body.message || "Não foi possível concluir a operação."));
  }
  return body;
}

async function refreshStatus() {
  const status = await api("/api/status");
  maxImages = status.max_images;
  let text = status.persons === 0
    ? "Nenhuma pessoa cadastrada"
    : `${status.persons} ${status.persons === 1 ? "pessoa cadastrada" : "pessoas cadastradas"}`;
  if (status.skipped_embeddings > 0) text += " · há cadastros de outro modelo que serão ignorados";
  $("status").textContent = text;
}

async function refreshPersons() {
  const persons = await api("/api/persons");
  const list = $("persons");
  list.replaceChildren(...persons.map(personRow));
  $("persons-empty").hidden = persons.length > 0;
}

function personRow(person) {
  const row = document.createElement("li");
  const info = document.createElement("div");
  const name = document.createElement("strong");
  name.textContent = person.name;
  const meta = document.createElement("small");
  const date = new Date(person.created_at).toLocaleDateString("pt-BR");
  meta.textContent = `Cadastro em ${date} ·${person.num_embeddings} ${person.num_embeddings === 1 ? "foto" : "fotos"}`;
  info.append(name, meta);

  const remove = document.createElement("button");
  remove.className = "danger";
  remove.textContent = "Excluir";
  let armed = false;
  remove.addEventListener("click", async () => {
    if (!armed) {
      armed = true;
      remove.textContent = "Confirmar exclusão definitiva";
      return;
    }
    remove.disabled = true;
    try {
      await api(`/api/persons/${encodeURIComponent(person.id)}`, { method: "DELETE" });
      showMessage($("persons-message"), "Pessoa e todos os dados associados foram excluídos.", true);
      await Promise.all([refreshPersons(), refreshStatus()]);
    } catch (error) {
      remove.disabled = false;
      showMessage($("persons-message"), error.message);
    }
  });
  row.append(info, remove);
  return row;
}

function selectTab(name) {
  for (const tab of ["recognize", "enroll", "persons"]) {
    const selected = tab === name;
    $(`tab-${tab}`).setAttribute("aria-selected", String(selected));
    $(`panel-${tab}`).hidden = !selected;
  }
  if (name !== "recognize") stopCamera();
  if (name !== "enroll") stopEnrollCamera();
  if (name === "persons") {
    hideMessage($("persons-message"));
    refreshPersons().catch((error) => showMessage($("persons-message"), error.message));
  }
}

function drawFaces(faces) {
  const canvas = $("canvas");
  const context = canvas.getContext("2d");
  const scale = canvas.width / (canvas.clientWidth || canvas.width);
  context.lineWidth = 3 * scale;
  context.font = `600 ${16 * scale}px system-ui, sans-serif`;
  context.textBaseline = "top";
  for (const face of faces) {
    const [x1, y1, x2, y2] = face.box;
    const color = face.recognized ? COLORS.ok : COLORS.bad;
    const label = face.recognized ? `${face.name} (${face.similarity.toFixed(2)})` : "Pessoa desconhecida";
    context.strokeStyle = color;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const height = 24 * scale;
    const width = context.measureText(label).width + 12 * scale;
    const labelY = y1 >= height ? y1 - height : y1;
    context.fillStyle = color;
    context.fillRect(x1, labelY, width, height);
    context.fillStyle = "#ffffff";
    context.fillText(label, x1 + 6 * scale, labelY + 4 * scale);
  }
}

function summarize(faces) {
  const result = $("result");
  result.classList.remove("ok", "bad");
  if (faces.length === 0) {
    result.textContent = "Nenhuma face na imagem";
    return;
  }
  const known = faces.filter((face) => face.recognized).map((face) => face.name);
  const unknown = faces.length - known.length;
  const parts = [...known];
  if (unknown > 0) parts.push(unknown === 1 ? "Pessoa desconhecida" : `${unknown} pessoas desconhecidas`);
  result.textContent = parts.join(" · ");
  result.classList.add(unknown === 0 ? "ok" : "bad");
}

function canvasBlob(canvas) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
}

async function recognize(blob) {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");
  return (await api("/api/recognize", { method: "POST", body: form })).faces;
}

async function cameraLoop() {
  const video = $("video");
  const canvas = $("canvas");
  const buffer = document.createElement("canvas");
  while (cameraStream) {
    if (video.videoWidth === 0) {
      await new Promise((resolve) => setTimeout(resolve, 50));
      continue;
    }
    const ratio = Math.min(1, MAX_FRAME_WIDTH / video.videoWidth);
    buffer.width = Math.round(video.videoWidth * ratio);
    buffer.height = Math.round(video.videoHeight * ratio);
    buffer.getContext("2d").drawImage(video, 0, 0, buffer.width, buffer.height);
    try {
      const faces = await recognize(await canvasBlob(buffer));
      if (!cameraStream) return;
      canvas.width = buffer.width;
      canvas.height = buffer.height;
      canvas.getContext("2d").drawImage(buffer, 0, 0);
      drawFaces(faces);
      summarize(faces);
      hideMessage($("recognize-message"));
    } catch (error) {
      showMessage($("recognize-message"), error.message);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
}

async function requestCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error(CAMERA_ERROR);
  try {
    return await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
  } catch {
    throw new Error(CAMERA_ERROR);
  }
}

function stopStream(stream) {
  stream?.getTracks().forEach((track) => track.stop());
}

function stopCamera() {
  stopStream(cameraStream);
  cameraStream = null;
  $("video").srcObject = null;
  $("camera-toggle").textContent = "Iniciar câmera";
}

async function toggleCamera() {
  if (cameraStream) {
    stopCamera();
    return;
  }
  hideMessage($("recognize-message"));
  try {
    cameraStream = await requestCamera();
  } catch (error) {
    showMessage($("recognize-message"), error.message);
    return;
  }
  $("video").srcObject = cameraStream;
  $("stage-hint").hidden = true;
  $("camera-toggle").textContent = "Parar câmera";
  cameraLoop();
}

async function recognizeFile(file) {
  stopCamera();
  hideMessage($("recognize-message"));
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const faces = await recognize(file);
    const canvas = $("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0);
    bitmap.close();
    $("stage-hint").hidden = true;
    drawFaces(faces);
    summarize(faces);
  } catch (error) {
    const message = error instanceof DOMException ? "O arquivo não é uma imagem válida." : error.message;
    showMessage($("recognize-message"), message);
  }
}

function renderPhotos() {
  const list = $("enroll-photos");
  list.querySelectorAll("img").forEach((image) => URL.revokeObjectURL(image.src));
  list.replaceChildren(...pendingPhotos.map((photo, index) => {
    const item = document.createElement("li");
    const image = document.createElement("img");
    image.src = URL.createObjectURL(photo);
    image.alt = `Foto ${index + 1}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remover foto ${index + 1}`);
    remove.addEventListener("click", () => {
      pendingPhotos.splice(index, 1);
      renderPhotos();
    });
    item.append(image, remove);
    return item;
  }));
}

function addPhotos(files) {
  const room = maxImages - pendingPhotos.length;
  if (files.length > room) showMessage($("enroll-message"), `Envie no máximo ${maxImages} fotos.`);
  pendingPhotos.push(...files.slice(0, Math.max(0, room)));
  renderPhotos();
}

function stopEnrollCamera() {
  stopStream(enrollStream);
  enrollStream = null;
  $("enroll-video").srcObject = null;
  $("enroll-video").hidden = true;
  $("enroll-capture").textContent = "Capturar da câmera";
}

async function captureForEnrollment() {
  hideMessage($("enroll-message"));
  const video = $("enroll-video");
  if (!enrollStream) {
    try {
      enrollStream = await requestCamera();
    } catch (error) {
      showMessage($("enroll-message"), error.message);
      return;
    }
    video.srcObject = enrollStream;
    video.hidden = false;
    $("enroll-capture").textContent = "Tirar foto";
    return;
  }
  const buffer = document.createElement("canvas");
  buffer.width = video.videoWidth;
  buffer.height = video.videoHeight;
  buffer.getContext("2d").drawImage(video, 0, 0);
  addPhotos([await canvasBlob(buffer)]);
}

async function submitEnrollment(event) {
  event.preventDefault();
  const message = $("enroll-message");
  hideMessage(message);
  const name = $("enroll-name").value.trim();
  if (!name) return showMessage(message, "Informe o nome da pessoa.");
  if (pendingPhotos.length === 0) return showMessage(message, "Adicione ao menos uma foto.");
  if (!$("enroll-consent").checked) return showMessage(message, "Confirme o consentimento da pessoa para continuar.");

  const form = new FormData();
  form.append("name", name);
  form.append("consent", "true");
  pendingPhotos.forEach((photo, index) => form.append("files", photo, `foto-${index}.jpg`));
  const submit = $("enroll-submit");
  submit.disabled = true;
  try {
    const person = await api("/api/persons", { method: "POST", body: form });
    showMessage(message, `Cadastro concluído: ${person.name}, ${person.num_embeddings} ${person.num_embeddings === 1 ? "foto" : "fotos"}.`, true);
    $("enroll-form").reset();
    pendingPhotos = [];
    renderPhotos();
    stopEnrollCamera();
    await refreshStatus();
  } catch (error) {
    showMessage(message, error.message);
  } finally {
    submit.disabled = false;
  }
}

for (const tab of ["recognize", "enroll", "persons"]) {
  $(`tab-${tab}`).addEventListener("click", () => selectTab(tab));
}
$("camera-toggle").addEventListener("click", toggleCamera);
$("recognize-file").addEventListener("change", (event) => {
  if (event.target.files[0]) recognizeFile(event.target.files[0]);
  event.target.value = "";
});
$("enroll-files").addEventListener("change", (event) => {
  hideMessage($("enroll-message"));
  addPhotos([...event.target.files]);
  event.target.value = "";
});
$("enroll-capture").addEventListener("click", captureForEnrollment);
$("enroll-form").addEventListener("submit", submitEnrollment);

refreshStatus().catch((error) => { $("status").textContent = error.message; });
