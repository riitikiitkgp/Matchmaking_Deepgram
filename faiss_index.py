import faiss
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
from utils import normalize_vector
from transcript_utils import extract_keywords

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["matchmaking"]
collection = db["transcripts"]
match_logs = db["match_logs"]

COOLDOWN_DAYS = 0

def build_faiss_index(vectors):
    if not vectors:
        return None
    vectors_np = np.array(vectors).astype("float32")
    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)
    return index

def is_in_cooldown(user_id, matched_id):
    log = match_logs.find_one({"user_id": user_id, "matched_id": matched_id})
    if log and "timestamp" in log:
        time_elapsed = datetime.utcnow() - log["timestamp"]
        return time_elapsed < timedelta(days=COOLDOWN_DAYS)
    return False

def log_match(user_id, matched_id):
    match_logs.update_one(
        {"user_id": user_id, "matched_id": matched_id},
        {"$set": {"timestamp": datetime.utcnow()}},
        upsert=True
    )
    match_logs.update_one(
        {"user_id": matched_id, "matched_id": user_id},
        {"$set": {"timestamp": datetime.utcnow()}},
        upsert=True
    )


def find_best_match_for_user(user_id, looking_for_gender):
    current_user = collection.find_one({"user_id": user_id})
    if not current_user or "embedding" not in current_user:
        return None

    current_vec = normalize_vector(np.array(current_user["embedding"], dtype=np.float32)).reshape(1, -1)
    current_keywords = extract_keywords(current_user.get("transcript", ""))

    # Get all candidates matching the gender and not the same user
    candidates = list(collection.find({
        "gender": looking_for_gender,
        "user_id": {"$ne": user_id}
    }))

    valid_vectors, valid_users, valid_transcripts, valid_keywords = [], [], [], []

    for cand in candidates:
        if "embedding" not in cand or is_in_cooldown(user_id, cand["user_id"]):
            continue
        vec = normalize_vector(np.array(cand["embedding"], dtype=np.float32))
        valid_vectors.append(vec)
        valid_users.append(cand["user_id"])
        valid_transcripts.append(cand.get("transcript", ""))
        valid_keywords.append(extract_keywords(cand.get("transcript", "")))

    if not valid_vectors:
        return None

    index = build_faiss_index(valid_vectors)
    if index is None:
        return None

    D, I = index.search(current_vec, 1)
    if I[0][0] == -1:
        return None

    matched_idx = I[0][0]
    matched_id = valid_users[matched_idx]
    similarity = float(D[0][0])

    # Get matched user details
    matched_user = collection.find_one({"user_id": matched_id})

    # Log both ways for mutual match
    log_match(user_id, matched_id)
    log_match(matched_id, user_id)

    # Find meaningful matched keywords (remove generic ones)
    matched_on = list(set(current_keywords) & set(valid_keywords[matched_idx]))
    matched_on = [kw for kw in matched_on if kw.lower() not in ["love", "like", "fun"]]

    match_reason = (
        f"You both share an interest in {', '.join(matched_on)}"
        if matched_on else "Your vibes match with each other"
    )

    return {
        "user_id": user_id,
        "your_transcript": current_user.get("transcript", ""),
        "matched_user_id": matched_id,
        "matched_transcript": matched_user.get("transcript", "") if matched_user else "",
        "similarity_score": similarity,
        "matched_on": matched_on,
        "match_reason": match_reason,
        "mutual_match": True
    }
