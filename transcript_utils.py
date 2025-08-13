# transcript_utils.py

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer

# Load small English NLP model for keyword extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load sentence transformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def transcript_to_embedding(transcript: str):
    """
    Convert transcript text to normalized embedding vector.
    """
    if not transcript.strip():
        return np.zeros(embedding_model.get_sentence_embedding_dimension()).tolist()
    embedding = embedding_model.encode(transcript, normalize_embeddings=True)
    return embedding.tolist()

STOP_KEYWORDS = {"love", "like", "hello", "hi", "hey", "yes", "my", "name", "also", "the", "a", "an", "i"}

def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = []
    for token in doc:
        if (
            token.is_alpha
            and token.text not in STOP_KEYWORDS
            and token.pos_ in {"NOUN", "VERB"}
        ):
            keywords.append(token.text)
    return list(set(keywords))

