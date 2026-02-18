# Hybrid Neural Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **Implementation of Neural Collaborative Filtering (NCF) and Content-Based Filtering for personalized item discovery.**

## Project Overview
This repository contains a production-grade implementation of a **Hybrid Recommender System**. The goal is to solve the "information overload" problem by predicting user preferences using two complementary strategies:

1.  **Neural Collaborative Filtering (NCF)**: A Deep Learning approach that learns low-dimensional embeddings of Users and Items to capture latent interaction patterns.
2.  **Content-Based Filtering**: A Vector Space Model (TF-IDF & Cosine Similarity) that recommends items based on metadata attributes.

> **Case Study**: This architecture is demonstrated using a **Media/Anime Dataset**, but the pipeline is designed to be domain-agnostic and applicable to e-commerce, streaming platforms, or retail.

## Key Architectural Features
* **Dual-Tower Architecture**: Combines user embeddings and item embeddings in a dense neural network layer.
* **Vectorization Pipeline**: Automated text processing (TF-IDF) to convert categorical metadata into mathematical vectors.
* **Embedding Layers**: Custom Keras layers to learn dense representations of sparse user-item interactions.
* **Scalability**: Designed to handle cold-start problems by falling back to content-based similarity when user history is sparse.

## Tech Stack
* **Core Logic**: Python, TensorFlow (Keras API), Scikit-Learn.
* **Data Manipulation**: Pandas, NumPy.
* **Visualization**: Matplotlib, Seaborn.

## Methodology
### 1. The "Similarity" Engine (Content-Based)
Calculates the geometric distance between item vectors.
* **Technique**: TF-IDF (Term Frequency-Inverse Document Frequency).
* **Metric**: Cosine Similarity.
* **Application**: "Users who viewed Item A might also like Item B (because they share similar tags)."

### 2. The "Predictive" Engine (Model-Based)
Predicts the probability of a user interacting with an item.
* **Model**: `RecommenderNet` (Subclassed from `keras.Model`).
* **Mechanism**:
    * Maps UserID and ItemID to Embedding Vectors.
    * Computes the dot product + bias.
    * Optimizes using **Adam** optimizer and **Binary Crossentropy** loss.
* **Application**: "Based on User A's past behavior, they have a 95% probability of liking Item Z."

## 📈 Performance
The model was trained on thousands of user interactions and evaluated using **Root Mean Squared Error (RMSE)**.
* **Validation RMSE**: *[Insert your RMSE here, e.g., 0.182]*
* **Precision@K**: The model successfully retrieves relevant items in the top-10 recommendations list.

## 💻 Usage
1.  **Clone the Repo**:
    ```bash
    git clone [https://github.com/adolesans/Hybrid-Neural-Recommendation-Engine.git](https://github.com/adolesans/Hybrid-Neural-Recommendation-Engine.git)
    ```
2.  **Run the Pipeline**:
    The main logic is encapsulated in the Jupyter Notebook.
    ```bash
    jupyter notebook notebook_recommendation.ipynb
    ```

## 📜 License
Distributed under the MIT License.

---
*Developed by [Annisa D.Y.](https://www.linkedin.com/in/annisa-dewiyanti)*
