from sentence_transformers import SentenceTransformer
from preprocessing import clean_arabic
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix



df = pd.read_csv("../Altibbi_Questions_Specialties (5).csv")

texts = df["question_body"].astype(str).tolist()
labels = df["specialty_name"].astype(str).tolist()
print(len(texts))
print(len(labels))
print(set(labels))

embedder = SentenceTransformer("intfloat/multilingual-e5-large")

texts_cleaned = [clean_arabic(t) for t in texts]
texts_prefixed = [f"passage: {t}" for t in texts_cleaned]

X = embedder.encode(
    texts_prefixed,
    batch_size=16,
    show_progress_bar=True
)
# X.shape → (n_samples, 1024)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(labels)  # shape (n_samples,)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

import joblib

joblib.dump(clf, "LR-disease_classifier.pkl")
joblib.dump(le, "LR-label_encoder.pkl")




from sklearn.svm import LinearSVC

clf_svm = LinearSVC(class_weight="balanced")
clf_svm.fit(X_train, y_train)


y_pred_svm = clf_svm.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_pred_svm,
    target_names=le.classes_
))

cm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



import joblib

joblib.dump(clf_svm, "SVM-disease_classifier.pkl")
joblib.dump(le, "SVM-label_encoder.pkl")
