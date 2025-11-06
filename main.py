# main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sqlite3
import base64, io, os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize models once (faster)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

DB = "faces.db"
THRESHOLD = 0.75  # higher = stricter

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    embedding TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

def image_from_b64(b64str):
    b64str = b64str.split(",")[-1]
    data = base64.b64decode(b64str)
    return Image.open(io.BytesIO(data)).convert("RGB")

def encoding_to_str(enc):
    return ",".join(map(str, enc.tolist()))

def str_to_encoding(s):
    return np.array(list(map(float, s.split(","))))

def get_embedding(img: Image.Image):
    """Return a 512-D embedding vector for the face in the image"""
    face = mtcnn(img)
    if face is None:
        return None
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device))
    return emb.squeeze(0).cpu().numpy()

class RegisterRequest(BaseModel):
    username: str
    image: str  # base64

class LoginRequest(BaseModel):
    image: str

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/register")
async def register(req: RegisterRequest):
    img = image_from_b64(req.image)
    emb = get_embedding(img)
    if emb is None:
        return JSONResponse({"error": "No face detected"}, status_code=400)

    emb_str = encoding_to_str(emb)
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, embedding) VALUES (?, ?)", (req.username, emb_str))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return JSONResponse({"error": "Username already exists"}, status_code=400)
    conn.close()
    return {"status": "registered", "username": req.username}

@app.post("/login-face")
async def login_face(req: LoginRequest):
    img = image_from_b64(req.image)
    emb = get_embedding(img)
    if emb is None:
        return JSONResponse({"error": "No face detected"}, status_code=400)

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, username, embedding FROM users")
    rows = c.fetchall()
    conn.close()

    best_score, best_user = -1, None
    for uid, username, emb_str in rows:
        db_emb = str_to_encoding(emb_str)
        dot = np.dot(emb, db_emb)
        denom = np.linalg.norm(emb) * np.linalg.norm(db_emb)
        cos = float(dot / denom)
        if cos > best_score:
            best_score, best_user = cos, {"id": uid, "username": username}

    if best_score >= THRESHOLD:
        return {"status": "matched", "user": best_user, "score": best_score}
    else:
        return JSONResponse({"status": "no_match", "best_score": best_score}, status_code=401)

@app.on_event("startup")
async def startup():
    init_db()
    print("Database initialized ✔")
