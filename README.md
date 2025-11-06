# 👁️ Face Login System (FastAPI + FaceNet)

A simple facial recognition login demo built with **FastAPI** and **facenet-pytorch**.

### 🧠 How it works
1. A user registers with a selfie (webcam capture).
2. The backend extracts a 512-D embedding from their face using a pretrained FaceNet CNN.
3. On login, a new image’s embedding is compared via cosine similarity.
4. If similarity ≥ threshold → authenticated ✅

### 🏗️ Tech Stack
- **Backend:** FastAPI
- **Face Embedding:** FaceNet (facenet-pytorch)
- **Frontend:** Simple HTML + JavaScript (Webcam API)
- **Database:** SQLite

### 🚀 Run locally
```bash
# create virtual env (Windows)
python -m venv venv
.\venv\Scripts\activate

# install deps
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install facenet-pytorch fastapi uvicorn numpy pillow

# run app
uvicorn main:app --reload
