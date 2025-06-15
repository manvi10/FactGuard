# 🛡️ FactGuard: Multi-Modal Fake News Detection Using NLP and Neural Networks

**FactGuard** is a full-stack web application that leverages state-of-the-art NLP models to detect fake news with high accuracy. It integrates a BERT-based deep learning model trained on real-world datasets, served through a backend API and accessed via a modern, interactive frontend.

---

## 🚀 Features

- 🧠 **BERT-based text classification** for fake news detection.
- 📚 Trained on **LIAR** benchmark dataset and additional **Kaggle fake news datasets**.
- 🔧 Robust **preprocessing pipeline** using Hugging Face Transformers.
- 🌐 **Frontend** built with React for real-time user interaction.
- 🖥️ **Backend API** using Flask or FastAPI to serve predictions.
- 📊 Performance metrics: **Accuracy, Precision, Recall, F1-score**, and **Confusion Matrix**.
- 🔄 Modular codebase for easy updates and extension to multi-modal data.

---

## 🧱 Project Structure
```plaintext
FactGuard/
├── data/ # LIAR & Kaggle datasets
├── models/ # Trained model files (BERT, etc.)
├── src/ # Scripts: preprocessing, training, evaluation
│ ├── preprocessing.py
│ ├── train_bert.py
│ ├── evaluate.py
│ └── utils.py
├── backend/ # Flask or FastAPI backend with API routes
├── frontend/ # React.js frontend interface
├── notebooks/ # Jupyter notebooks for EDA and experimentation
└── README.md
```

---

## ⚙️ Technologies Used

| Layer        | Tools / Libraries                               	|
|--------------|----------------------------------------------------|
| **Modeling** | BERT, Hugging Face Transformers, TensorFlow/Keras 	|
| **Backend**  | Flask / FastAPI                                  	|
| **Frontend** | React.js, Axios, HTML5/CSS3, Tailwind            	|
| **Data**     | LIAR Dataset, Kaggle Fake News Dataset           	|
| **Visualization** | Matplotlib, Seaborn, Scikit-learn Metrics     |

---

## 📁 Datasets Used
LIAR dataset — Labelled fake/real statements from politifact.com

Kaggle Fake News dataset — Real vs fake news articles

---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or improve.

---

## 📬 Contact
Sai Manvi Pallapothu
Email: manvipallapothu31@gmail.com
GitHub: https://github.com/manvi10
