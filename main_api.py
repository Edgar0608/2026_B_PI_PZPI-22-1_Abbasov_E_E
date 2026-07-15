"""
PoC API server: приймає ескіз + prompt, повертає згенероване зображення.
Запускається в Google Colab поряд з ComfyUI.

ComfyUI : 127.0.0.1:8188  (не відкривається назовні)
FastAPI : 0.0.0.0:8000    (відкривається через cloudflared)

Модель    : v1-5-pruned-emaonly.ckpt
ControlNet: control_v11p_sd15_scribble.pth  (якщо передано ескіз)

ЗМІНИ v3:
- Авторизація через JWT (register / login)
- SQLite база зберігається на Google Drive
- /generate і /result захищені токеном
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import json
import uuid
import urllib.request
import urllib.parse
import websocket          # pip install websocket-client
import time
import threading
import sqlite3
import hashlib
import hmac
import base64
from enum import Enum
from typing import Optional
from datetime import datetime, timedelta

app = FastAPI(title="Sketch → Image PoC v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COMFYUI_URL = "127.0.0.1:8188"
HTML_PATH   = "/content/drive/MyDrive/AI_project/AI_Project/index.html"

# ── Model presets ─────────────────────────────────────────────────────
MODEL_PRESETS = {
    "anime": {
        "label":      "Anime / Illustration (Anything v5)",
        "checkpoint": "anything-v5.safetensors",
        "controlnet": "control_v11p_sd15_scribble_fp16.safetensors",
        "vae":        "vae-ft-mse-840000-ema-pruned.safetensors",
        "resolutions": ["512x512", "512x768"],
    },
    "orangemix": {
        "label":      "Stylized / Dark (OrangeMix AOM3)",
        "checkpoint": "AOM3A3_orangemixs.safetensors",
        "controlnet": "control_v11p_sd15_scribble_fp16.safetensors",
        "vae":        "vae-ft-mse-840000-ema-pruned.safetensors",
        "resolutions": ["512x512", "512x768"],
    },
    "realistic": {
        "label":      "Realistic Photo (Realistic Vision v6)",
        "checkpoint": "Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
        "controlnet": "control_v11p_sd15_scribble_fp16.safetensors",
        "vae":        "vae-ft-mse-840000-ema-pruned.safetensors",
        "resolutions": ["512x768", "768x512", "640x640"],
    },
}
DEFAULT_PRESET = "anime"

# ── БД на Google Drive (не губиться при перезапуску Colab) ───────────
DB_PATH  = "/content/drive/MyDrive/AI_project/AI_Project/users.db"
JWT_SECRET = "temp-key-12345-change-later"   # Поміняй на щось складне!
JWT_EXPIRE_HOURS = 24 * 365 * 10   # токен живе 10 років (фактично назавжди)

# ── SQLite helpers ────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email    TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created  TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL,
            job_id           TEXT NOT NULL,
            title            TEXT,
            prompt           TEXT NOT NULL,
            negative_prompt  TEXT,
            model_preset     TEXT,
            resolution       TEXT,
            sketch_data      BLOB,
            result_data      BLOB,
            created          TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def migrate_db():
    conn = get_db()
    existing = [r[1] for r in conn.execute("PRAGMA table_info(generations)").fetchall()]
    needed = {
        "title":           "TEXT",
        "sketch_data":     "BLOB",
        "result_data":     "BLOB",
        "negative_prompt": "TEXT",
        "model_preset":    "TEXT",
        "resolution":      "TEXT",
        "user_id":         "INTEGER",
        "job_id":          "TEXT",
        "steps":           "INTEGER",
    }
    for col, col_type in needed.items():
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE generations ADD COLUMN {col} {col_type}")
                print(f"Migration: added column {col}")
            except Exception as e:
                print(f"Migration skip {col}: {e}")
    conn.commit()
    conn.close()

migrate_db()

# ── Password helpers ──────────────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = uuid.uuid4().hex
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${h}"

def verify_password(password: str, stored: str) -> bool:
    salt, h = stored.split("$", 1)
    return hmac.compare_digest(h, hashlib.sha256((salt + password).encode()).hexdigest())

# ── JWT helpers (без зовнішніх бібліотек) ────────────────────────────

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * pad)

def create_jwt(user_id: int, username: str) -> str:
    header  = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    exp     = int((datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)).timestamp())
    payload = _b64url(json.dumps({"sub": user_id, "username": username, "exp": exp}).encode())
    sig     = _b64url(
        hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{sig}"

def decode_jwt(token: str) -> dict:
    try:
        header, payload, sig = token.split(".")
        expected_sig = _b64url(
            hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig, expected_sig):
            raise ValueError("bad signature")
        data = json.loads(_b64url_decode(payload))
        if data["exp"] < int(datetime.utcnow().timestamp()):
            raise ValueError("token expired")
        return data
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")

# ── Auth dependency ───────────────────────────────────────────────────

bearer = HTTPBearer()

def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    return decode_jwt(creds.credentials)

# ── In-memory job store ───────────────────────────────────────────────
jobs: dict = {}
jobs_lock = threading.Lock()

class JobStatus(str, Enum):
    pending = "pending"
    done    = "done"
    error   = "error"

# ── ComfyUI helpers ───────────────────────────────────────────────────

def _post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        f"http://{COMFYUI_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def upload_image_to_comfy(image_bytes: bytes, filename: str) -> str:
    """Upload image to ComfyUI. Converts to plain RGB PNG to avoid segfault in video_types.py."""
    import io
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to plain RGB PNG — this avoids the segfault in LoadImage/video_types.py
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        image_bytes = buf.getvalue()
        filename = filename.rsplit(".", 1)[0] + ".png"
    except Exception as e:
        # If PIL fails, upload as-is
        print(f"[WARN] PIL convert failed: {e}, uploading raw bytes")

    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"http://{COMFYUI_URL}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["name"]

def queue_prompt(workflow: dict, client_id: str) -> str:
    resp = _post_json("/prompt", {"prompt": workflow, "client_id": client_id})
    return resp["prompt_id"]

def get_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    params = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": folder_type}
    )
    with urllib.request.urlopen(f"http://{COMFYUI_URL}/view?{params}", timeout=60) as r:
        return r.read()

def wait_for_image(client_id: str, prompt_id: str, timeout: int = 300) -> bytes:
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFYUI_URL}/ws?clientId={client_id}", timeout=timeout)
    ws.settimeout(timeout)
    try:
        while True:
            out = ws.recv()
            if not isinstance(out, str):
                continue
            msg   = json.loads(out)
            mtype = msg.get("type")
            data  = msg.get("data", {})
            if mtype == "execution_error" and data.get("prompt_id") == prompt_id:
                raise RuntimeError(f"ComfyUI execution error: {data}")
            if mtype == "executing":
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break
    finally:
        ws.close()

    with urllib.request.urlopen(
        f"http://{COMFYUI_URL}/history/{prompt_id}", timeout=30
    ) as r:
        history = json.loads(r.read())

    outputs = history.get(prompt_id, {}).get("outputs", {})
    for _, node_output in outputs.items():
        if node_output.get("images"):
            img = node_output["images"][0]
            return get_image(img["filename"], img["subfolder"], img["type"])
    raise RuntimeError("No image found in ComfyUI history")

# ── Workflow builder ──────────────────────────────────────────────────

BASE_NEGATIVE = (
    "text, watermark, signature, low quality, blurry, "
    "bad anatomy, extra limbs, deformed, ugly, noise"
)

def build_workflow(
    prompt_text: str,
    sketch_filename: Optional[str],
    negative_prompt: str = "",
    preset_key: str = DEFAULT_PRESET,
    resolution: str = "512x512",
    steps: int = 30,
) -> dict:
    preset = MODEL_PRESETS.get(preset_key, MODEL_PRESETS[DEFAULT_PRESET])

    # Validate & parse resolution
    allowed = preset["resolutions"]
    if resolution not in allowed:
        resolution = allowed[0]
    width, height = (int(x) for x in resolution.split("x"))

    # Combine negative prompts
    negative = BASE_NEGATIVE
    if negative_prompt.strip():
        negative = negative_prompt.strip() + ", " + negative
    seed = int(time.time() * 1000) & 0xFFFF_FFFF

    # Clamp steps to a safe range (slider allows 0-100, but KSampler needs >=1)
    steps = max(1, min(100, int(steps)))

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": preset["checkpoint"]}},
        "2": {"class_type": "CLIPTextEncode",          "inputs": {"text": prompt_text, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",          "inputs": {"text": negative,    "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",         "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": steps, "cfg": 7.5,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": preset["vae"]}},
        "7": {"class_type": "VAEDecode",  "inputs": {"samples": ["5", 0], "vae": ["6", 0]}},
        "8": {"class_type": "SaveImage",  "inputs": {"filename_prefix": "poc", "images": ["7", 0]}},
    }

    if sketch_filename:
        wf["9"]  = {"class_type": "LoadImage",        "inputs": {"image": sketch_filename}}
        wf["10"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": preset["controlnet"]}}
        wf["11"] = {"class_type": "ControlNetApply",
                    "inputs": {"strength": 0.9, "conditioning": ["2", 0], "control_net": ["10", 0], "image": ["9", 0]}}
        wf["12"] = {"class_type": "ControlNetApply",
                    "inputs": {"strength": 0.9, "conditioning": ["3", 0], "control_net": ["10", 0], "image": ["9", 0]}}
        wf["5"]["inputs"]["positive"] = ["11", 0]
        wf["5"]["inputs"]["negative"] = ["12", 0]

    return wf

# ── Background worker ─────────────────────────────────────────────────

def run_generation(job_id: str, prompt_text: str, sketch_bytes: Optional[bytes], negative_prompt: str = "", preset_key: str = DEFAULT_PRESET, resolution: str = "512x512", title: str = "", steps: int = 30):
    client_id = str(uuid.uuid4())
    sketch_filename = None
    try:
        if sketch_bytes:
            safe_name = f"sketch_{client_id}.png"
            sketch_filename = upload_image_to_comfy(sketch_bytes, safe_name)
        workflow  = build_workflow(prompt_text, sketch_filename, negative_prompt, preset_key, resolution, steps)
        prompt_id = queue_prompt(workflow, client_id)
        image_data = wait_for_image(client_id, prompt_id)
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.done
            jobs[job_id]["image"]  = image_data
        # Save to DB
        job = jobs.get(job_id, {})
        conn = get_db()
        try:
            conn.execute(
                """INSERT INTO generations
                   (user_id, job_id, title, prompt, negative_prompt, model_preset, resolution, steps, sketch_data, result_data, created)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.get("user_id"), job_id, title, prompt_text, negative_prompt,
                    preset_key, resolution, steps,
                    sketch_bytes, image_data,
                    datetime.utcnow().isoformat()
                )
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.error
            jobs[job_id]["error"]  = str(e)

# ── Auth endpoints ────────────────────────────────────────────────────

@app.post("/auth/register")
def register(
    username: str = Form(...),
    email:    str = Form(...),
    password: str = Form(...),
):
    if len(username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if "@" not in email:
        raise HTTPException(400, "Invalid email")

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password, created) VALUES (?, ?, ?, ?)",
            (username.strip(), email.strip().lower(), hash_password(password), datetime.utcnow().isoformat())
        )
        conn.commit()
        row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        token = create_jwt(row["id"], username)
        return {"token": token, "username": username}
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Username or email already taken")
    finally:
        conn.close()


@app.post("/auth/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, username, password FROM users WHERE username = ?",
            (username.strip(),)
        ).fetchone()
    finally:
        conn.close()

    if row is None or not verify_password(password, row["password"]):
        raise HTTPException(401, "Invalid username or password")

    token = create_jwt(row["id"], row["username"])
    return {"token": token, "username": row["username"]}


@app.get("/auth/me")
def me(user: dict = Depends(require_auth)):
    return {"user_id": user["sub"], "username": user["username"]}


# ── Main endpoints (захищені токеном) ─────────────────────────────────

@app.get("/models")
def list_models():
    """Return available model presets for the frontend selector."""
    return {
        key: {"label": v["label"], "resolutions": v["resolutions"]}
        for key, v in MODEL_PRESETS.items()
    }


@app.get("/")
def index():
    return FileResponse(HTML_PATH, media_type="text/html")

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    title: str = Form(...),
    negative_prompt: str = Form(""),
    model_preset: str = Form(DEFAULT_PRESET),
    resolution: str = Form("512x512"),
    steps: int = Form(30),
    sketch: UploadFile | None = File(None),
    user: dict = Depends(require_auth),
):
    if not prompt.strip():
        raise HTTPException(400, "prompt is empty")
    if not title.strip():
        raise HTTPException(400, "title is empty")
    if model_preset not in MODEL_PRESETS:
        raise HTTPException(400, f"Unknown model preset: {model_preset}")

    # Validate resolution against preset
    allowed = MODEL_PRESETS[model_preset]["resolutions"]
    if resolution not in allowed:
        resolution = allowed[0]

    # Clamp steps to a sane range (UI slider is 0-100, KSampler needs at least 1)
    steps = max(1, min(100, steps))

    sketch_bytes = None
    if sketch is not None:
        sketch_bytes = await sketch.read()
        if not sketch_bytes:
            raise HTTPException(400, "empty sketch file")

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": JobStatus.pending, "image": None, "error": None, "user_id": user["sub"]}

    threading.Thread(
        target=run_generation,
        args=(job_id, prompt.strip(), sketch_bytes, negative_prompt.strip(), model_preset, resolution, title.strip(), steps),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/result/{job_id}")
def get_result(job_id: str, user: dict = Depends(require_auth)):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(404, "job not found")

    if job["status"] == JobStatus.pending:
        return {"status": "pending"}

    if job["status"] == JobStatus.error:
        return {"status": "error", "detail": job["error"]}

    image_data = job["image"]
    with jobs_lock:
        del jobs[job_id]
    return Response(content=image_data, media_type="image/png")


@app.get("/history")
def get_history(
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None,
    search:    Optional[str] = None,
    sort:      str = "date_desc",
    user: dict = Depends(require_auth),
):
    conn = get_db()
    try:
        query  = "SELECT id, job_id, title, prompt, negative_prompt, model_preset, resolution, steps, created FROM generations WHERE user_id = ?"
        params = [user["sub"]]
        if date_from:
            query  += " AND created >= ?"
            params.append(date_from)
        if date_to:
            query  += " AND created <= ?"
            params.append(date_to + "T23:59:59")
        if search and search.strip():
            query  += " AND LOWER(COALESCE(title, '')) LIKE ?"
            params.append(f"%{search.strip().lower()}%")

        sort_map = {
            "date_desc":  "created DESC",
            "date_asc":   "created ASC",
            "title_asc":  "title COLLATE NOCASE ASC, created DESC",
            "title_desc": "title COLLATE NOCASE DESC, created DESC",
        }
        query += " ORDER BY " + sort_map.get(sort, "created DESC")

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/history/{gen_id}/sketch")
def get_history_sketch(gen_id: int, user: dict = Depends(require_auth)):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT sketch_data FROM generations WHERE id = ? AND user_id = ?",
            (gen_id, user["sub"])
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["sketch_data"] is None:
        raise HTTPException(404, "not found")
    return Response(content=row["sketch_data"], media_type="image/png")


@app.get("/history/{gen_id}/result")
def get_history_result(gen_id: int, user: dict = Depends(require_auth)):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT result_data FROM generations WHERE id = ? AND user_id = ?",
            (gen_id, user["sub"])
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["result_data"] is None:
        raise HTTPException(404, "not found")
    return Response(content=row["result_data"], media_type="image/png")


@app.delete("/history/{gen_id}")
def delete_history(gen_id: int, user: dict = Depends(require_auth)):
    conn = get_db()
    try:
        conn.execute(
            "DELETE FROM generations WHERE id = ? AND user_id = ?",
            (gen_id, user["sub"])
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")