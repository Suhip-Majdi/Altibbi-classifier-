🩺 Altibbi Disease Classifier

A multi-class disease classification system that predicts disease names from medical text using a combination of traditional Machine Learning models and modern semantic embedding techniques.
The project explores and compares multiple classifiers and embedding methods to achieve robust and accurate medical text classification across multilingual datasets.

🚀 Features
Multi-class disease classification from medical text
Support for multilingual medical input
Comparison between traditional ML and semantic embedding approaches
Uses both statistical and transformer-based text representations
End-to-end preprocessing, embedding generation, training, and evaluation pipeline
🧠 Models Used

This project evaluates several Machine Learning algorithms, including:

Logistic Regression
Support Vector Machines (SVMs)
Decision Trees
Neural Networks
📚 Embedding Techniques

The system leverages multiple text representation methods:

Traditional Embeddings
TF-IDF
Word2Vec
Transformer-Based Embeddings
NTFloat/multilingual-e5-large from Hugging Face

These embeddings help capture both syntactic and semantic relationships in medical text for improved classification performance.

🏥 Project Goal

The main objective of this project is to build an intelligent disease classification system capable of:

Identifying disease names from medical descriptions
Handling multilingual medical text
Improving semantic understanding using advanced embeddings
Comparing classical ML methods with deep semantic representations
🛠️ Tech Stack
Python
Scikit-learn
TensorFlow / Keras
Hugging Face Transformers
Gensim
Pandas
NumPy
📂 Project Structure
Altibbi-classifier/
│
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks
├── models/                 # Saved trained models
├── embeddings/             # Generated embeddings
├── preprocessing/          # Text preprocessing scripts
├── training/               # Training scripts
├── evaluation/             # Evaluation metrics and results
├── requirements.txt
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/Altibbi-classifier.git
cd Altibbi-classifier

Install dependencies:

pip install -r requirements.txt
▶️ Usage

Run the training pipeline:

python train.py

Run inference:

python predict.py
📊 Evaluation Metrics

The project evaluates model performance using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
🌍 Multilingual Support

The use of the NTFloat/multilingual-e5-large embedding model enables strong multilingual semantic understanding, making the classifier suitable for handling medical text in multiple languages, including Arabic and English.

🔬 Future Improvements
Fine-tuning transformer models on medical datasets
Deploying the model as an API
Building a medical chatbot integration
Adding explainability and attention visualization
Expanding disease categories and datasets
🤝 Contributing

Contributions are welcome!
Feel free to fork the repository, create a feature branch, and submit a pull request.

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Developed by Suhip Majdi
Data Science & Artificial Intelligence Engineer
