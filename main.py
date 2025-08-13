import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel
import numpy as np
import io
import uvicorn
from transcript_utils import transcript_to_embedding, extract_keywords
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from faiss_index import find_best_match_for_user
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["matchmaking"]
collection = db["transcripts"]
match_logs = db["match_logs"]

# Deepgram API Key
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

def transcribe_with_deepgram(audio_bytes: bytes) -> str:
    try:
        payload: FileSource = { "buffer": audio_bytes }

        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            language="en-US"
        )

        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

@app.post("/upload_audio/")
async def upload_audio(
    user_id: str = Form(...),
    audio: UploadFile = File(...)
):
    try:
        audio_bytes = await audio.read()
        transcript = transcribe_with_deepgram(audio_bytes)
        embedding = transcript_to_embedding(transcript)

        collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "transcript": transcript,
                    "embedding": embedding
                }
            },
            upsert=True
        )

        return {
            "user_id": user_id,
            "transcript": transcript
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/match_user/{user_id}")
def match_user(user_id: str, looking_for_gender: str):
    try:
        match_data = find_best_match_for_user(user_id, looking_for_gender)
        if match_data is None:
            raise HTTPException(status_code=404, detail="No suitable match found")
        return match_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
