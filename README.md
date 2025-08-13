# Voice-Based Matchmaking API

FastAPI service that:
- Transcribes uploaded audio with Deepgram
- Embeds transcripts and stores them in MongoDB
- Finds the best match between users using FAISS cosine-similarity (inner product over normalized vectors)

## Overview

**Main components**
- `main.py` – FastAPI app exposing `/upload_audio/` and `/match_user/{user_id}`
- `faiss_index.py` – FAISS index + matching logic
- `utils.py` – helper utilities (e.g., vector normalization)
- `requirements.txt` – core dependency pinned for Deepgram SDK
- `.env` – holds `DEEPGRAM_API_KEY`
- `transcript_utils.py` – **you must provide this module** (see below)

MongoDB is used to store user transcripts, embeddings, and match logs.

---

## Prerequisites

- **Python 3.10+**
- **MongoDB** running locally on `mongodb://localhost:27017`
- **Deepgram account & API key** (Model used: `nova-3`)
- Ability to install Python dependencies (pip/venv recommended)

> ⚠️ Never commit your real `.env` to version control. Put it in `.gitignore`.

---

## Installation

# 1) Create and activate a virtual environment (recommended)
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\\Scripts\\Activate.ps1

# 2) Upgrade pip
pip install --upgrade pip

# 3) Install dependencies

pip install -r requirements.txt
# plus the libraries imported by the code
pip install fastapi pymongo pydantic numpy uvicorn faiss-cpu python-dotenv
If faiss-cpu fails on your platform, try:

Linux/macOS: pip install faiss-cpu

Windows (alternate): pip install faiss-cpu-windows (community builds) or use WSL.
