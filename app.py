import time
import joblib
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing import clean_arabic
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ---------- Database ----------
conn = psycopg2.connect(
    host="localhost",
    database="altibbi_db",
    user="postgres",
    password="2003",
    port=5432
)
cursor = conn.cursor()


clf = joblib.load("SVM-disease_classifier.pkl")
le = joblib.load("svm-label_encoder.pkl")
embedder = SentenceTransformer("intfloat/multilingual-e5-large")


class Question(BaseModel):
    text: str



def predict_disease(question):
    start_time = time.time()

    cleaned = clean_arabic(question)
    formatted = f"query: {cleaned}"

    embedding = embedder.encode([formatted])
    pred_class = clf.predict(embedding)[0]
    disease = le.inverse_transform([pred_class])[0]

    end_time = time.time()
    elapsed = end_time - start_time


    cursor.execute("""
        INSERT INTO predictions (question_body, specialty_name)
        VALUES (%s, %s)
    """, (question, disease))

    conn.commit()

    return disease, elapsed

@app.post("/predict")
def predict(data: Question):
    disease, time_taken = predict_disease(data.text)

    return {
        "question": data.text,
        "prediction": disease,
        "inference_time_seconds": round(time_taken, 4)
    }
