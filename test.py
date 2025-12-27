import time
import joblib
from preprocessing import clean_arabic
from sentence_transformers import SentenceTransformer
import psycopg2


# clf = joblib.load("SVM-disease_classifier.pkl")
# le = joblib.load("SVM-label_encoder.pkl")

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


def predict_disease(question):
    start_time = time.time()

    question = clean_arabic(question)
    question = f"query: {question}"

    embedding = embedder.encode([question])
    pred_class = clf.predict(embedding)[0]

    disease = le.inverse_transform([pred_class])[0]

    end_time = time.time()
    elapsed_time = end_time - start_time

    cursor.execute(
        """
        INSERT INTO predictions (question_body, specialty_name)
        VALUES (%s, %s)
        """,
        (question, disease)
    )

    conn.commit()

    return disease, elapsed_time

# question= "اعاني من الم في الصدر وضيق في التنفس"
while True:
    question = str(input("what are you suffering from ?"))
    prediction, time_taken = predict_disease(question)
    print("Prediction:", prediction)
    if question == 'q' or question == 'Q':
        print("Exiting...")
        break

    print(f"Inference time: {time_taken:.4f} seconds")




# import time
#
# def predict_disease(question):
#     t0 = time.time()
#
#     question = clean_arabic(question)
#     t1 = time.time()
#
#     question = f"query: {question}"
#     embedding = embedder.encode([question])
#     t2 = time.time()
#
#     pred_class = clf.predict(embedding)[0]
#     disease = le.inverse_transform([pred_class])[0]
#     t3 = time.time()
#
#     return {
#         "prediction": disease,
#         "cleaning_time": t1 - t0,
#         "embedding_time": t2 - t1,
#         "classification_time": t3 - t2,
#         "total_time": t3 - t0
#     }
#
#
# result = predict_disease("اعاني من الم في الصدر وضيق في التنفس")
#
# print("Prediction:", result["prediction"])
# print(f"Cleaning time: {result['cleaning_time']:.4f}s")
# print(f"Embedding time: {result['embedding_time']:.4f}s")
# print(f"Classification time: {result['classification_time']:.4f}s")
# print(f"Total inference time: {result['total_time']:.4f}s")
