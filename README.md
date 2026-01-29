# Explainable-Clinical-Relation-Extraction-from-Medical-Text-Using-Interpretable-NLP-Models

# 📌 Project Overview

This project focuses on clinical relation extraction from medical text using a combination of traditional machine learning, deep learning, transformer-based models, and a hybrid architecture.
The goal is to accurately identify and classify semantic relationships between clinical entities while maintaining interpretability and robustness, which are critical in healthcare applications.

# 🎯 Objectives

- Extract meaningful clinical relationships from unstructured medical text

- Compare ML, DL, Transformer, and Hybrid models

- Improve performance using a hybrid architecture

- Ensure model interpretability and generalization

# 📂 Dataset Used

- MODI 2021 Clinical Dataset (P2 Dataset)

- Domain: Medical / Clinical Text

- Task: Relation extraction between medical entities

- Preprocessing includes:

 1. **Text normalization**

  2. **Tokenization**

  3.  **Padding & sequencing**

 4. **Feature extraction (TF-IDF & embeddings)**

# ⚙️ Models Implemented

🔹 Traditional Machine Learning Models

- Logistic Regression

- Decision Tree

- Random Forest

🔹 Deep Learning Models

- Simple RNN

- LSTM (Long Short-Term Memory)

- GRU (Gated Recurrent Unit)

🔹 Transformer-Based Model

- Transformer (Attention-based architecture)

🔹 Hybrid Model (Proposed)

- Transformer + GRU + Random Forest

- Transformer: Contextual embedding extraction

- GRU: Sequential pattern learning

-Random Forest: Final classification for stability & interpretability

# 📊 Performance Metrics

Each model is evaluated using:

- Test Accuracy

- Macro Average F1-Score

- Weighted Average F1-Score

- Training Accuracy

- Training Log Loss

- Test Log Loss

# 📊 Results

🔹 Model Performance Comparison

| Model | Test Accuracy | Macro Avg F1 | Weighted Avg F1 | Training Accuracy | Train Log Loss | Test Log Loss |
|------|--------------|--------------|-----------------|------------------|---------------|--------------|
| Logistic Regression | 0.8633 | 0.7216 | 0.8603 | 0.8895 | 0.4924 | 0.5430 |
| Decision Tree | 0.8358 | 0.7802 | 0.8348 | 0.9635 | 0.0782 | 5.0890 |
| Random Forest | 0.8564 | 0.7288 | 0.8549 | 0.9635 | 0.1552 | 0.8611 |
| Simple RNN | 0.7292 | 0.4962 | 0.7138 | 0.7914 | 0.7256 | 0.9667 |
| LSTM | 0.9539 | 0.8181 | 0.9541 | 0.9908 | 0.0289 | 0.2309 |
| GRU | 0.9564 | 0.8535 | 0.9566 | 0.9933 | 0.0175 | 0.2475 |
| Transformer | 0.8966 | 0.8763 | 0.8966 | 0.9659 | 0.0985 | 0.8108 |
| **Hybrid (Transformer + GRU + RF)** | **0.9616** | **0.8314** | **0.9616** | **0.9951** | **0.0210** | **0.2476** |

# 🏆 Key Findings

- Traditional ML models perform well on structured features but lack contextual understanding.

- Simple RNN underperforms due to vanishing gradient issues.

- LSTM and GRU significantly improve sequence modeling.

- Transformer effectively captures global contextual dependencies using attention.

- ✅ Hybrid model achieves the best overall performance (96.16% test accuracy), combining:

 1. Context awareness

2.  Sequential learning

3. Robust ensemble-based classification

# 📉 Visualization

📈 Accuracy Comparison Graph

### Model Accuracy Comparison

![Model Performance Comparison](results/model_training_accuracy_comparison.png)

### Model Accuracy Comparison

![Model Performance Comparison](results/model_testing_accuracy_comparison.png)

### Model Loss Comparison

![Model Performance Comparison](results/model_training_loss_comparison.png)

### Model Loss Comparison

![Model Performance Comparison](results/model_testing_loss_comparison.png)



# 🧠 Why Hybrid Architecture?

- Transformers provide semantic context

- GRU captures temporal dependencies

- Random Forest improves decision stability & interpretability

- Hybrid design reduces overfitting and enhances generalization

# 🚀 Applications

- Clinical decision support systems

- Medical knowledge graph construction

- Electronic Health Record (EHR) analysis

- Explainable AI in healthcare

- Biomedical text mining

# 🛠️ Tech Stack

- Python

- Scikit-learn

- TensorFlow / Keras

- Pandas, NumPy

- Matplotlib / Seaborn

- Google Colab

# 📌 Conclusion

This project demonstrates that combining Transformer-based embeddings, GRU sequence modeling, and Random Forest classification significantly enhances clinical relation extraction performance while maintaining interpretability—making it suitable for real-world healthcare AI applications.

# 📎 Future Work

- Integration with large-scale biomedical knowledge bases

- Incorporation of domain-specific transformers (BioBERT, ClinicalBERT)

- Explainability using SHAP / attention visualization

- Deployment as a clinical NLP service
