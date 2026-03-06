# Ethereum Fraud Detection System

## A Hybrid Machine Learning & Deep Learning Approach for Detecting Illicit Activities on the Ethereum Blockchain

**Academic Project – Master 2 PFM**  
**Authors:** Mouaad Es-safryouy, Ilyass Bouchtaoui

---

## Abstract

Blockchain systems such as Ethereum provide transparency and decentralization, but their pseudo-anonymous nature also facilitates illicit activities including money laundering, phishing attacks, and Ponzi schemes.

This project proposes an advanced **fraud detection system** that integrates **Gradient Boosting (CatBoost)** and **Deep Learning (Artificial Neural Networks, ANN)** into a high-performance **Hybrid Ensemble Model**.

The solution relies on:

- Extensive **feature engineering**
- Specialized **data augmentation (Feature Jittering)**
- A **hybrid ensemble architecture**
- **Threshold Optimization**

By combining machine learning and deep learning predictions, the system achieved strong performance:

- **Accuracy:** 94.16%
- **F1-Score:** 0.9365
- **ROC-AUC:** 0.9864

To ensure the model learns **behavioral fraud patterns rather than memorizing historical data**, robustness was evaluated using **synthetically generated transaction scenarios**. Experimental results demonstrate that **combining gradient boosting with deep learning significantly improves fraud detection performance and generalization**.

---

## 1. Introduction

The rapid growth of blockchain-based financial systems has introduced new challenges in **transaction monitoring and fraud detection**.

Unlike traditional banking systems, Ethereum does not rely on centralized identity verification, making it difficult to detect fraudulent behavior at the wallet address level.

Machine learning techniques offer a promising solution by analyzing **transaction dynamics and behavioral patterns**. However, a key challenge is ensuring models **generalize to unseen fraud strategies rather than overfitting known addresses**.

This project addresses this challenge by combining:

- Classical machine learning models
- Deep learning (ANN)
- Hybrid ensemble learning
- Synthetic data validation

to build a **robust and generalizable fraud detection system**.

---

## 2. Objectives

The main objectives of this project are:

- Design a **complete ML and DL pipeline** for Ethereum fraud detection
- Extract meaningful **financial and temporal features** from blockchain data
- Implement **data augmentation using Feature Jittering** to prevent memorization
- Benchmark multiple **state-of-the-art ML models**
- Implement a **deep learning ANN model** for complex pattern learning
- Combine models using a **Hybrid Ensemble (CatBoost + ANN)**
- Optimize classification using **Threshold Optimization**
- Validate generalization using **synthetic transaction scenarios**

---

## 3. Data Processing and Feature Engineering

Raw Ethereum transaction logs were processed to construct a structured dataset at the **wallet address level**.

Feature engineering captured both **financial characteristics** and **temporal transaction patterns**:

- Transaction frequency and inter-arrival times
- Total incoming and outgoing transaction values
- Account lifetime and activity duration
- Balance-related statistics and variability measures

These features reflect **behavioral patterns commonly associated with fraudulent activity**.

---

### 3.1 Data Augmentation & Normalization

To improve generalization and reduce overfitting:

**Log Transformation:** Applied to 22 financial features to reduce skewed distributions.  

**Feature Jittering:** Generated synthetic fraud "clones" by injecting 1% Gaussian noise (2 clones per fraudster), forcing the model to learn **behavioral concepts** rather than exact values.  

**MinMax Scaling:** All features normalized to `[0, 1]` to optimize convergence for ML and DL models.

---

## 4. Methodology

### 4.1 Base Models (Level 0)

Several classifiers were selected to capture complementary perspectives:

- **XGBoost:** Gradient boosting decision trees optimized for tabular data. High individual performance.
- **LightGBM:** Fast and memory-efficient gradient boosting.
- **CatBoost:** Handles categorical data and complex patterns with minimal tuning.
- **ExtraTrees Classifier:** Extremely randomized trees for variance reduction.
- **Random Forest:** Ensemble baseline model.
- **SVM:** Traditional baseline classifier.
- **TabNet:** Attention-based deep learning architecture for tabular data.

---

### 4.2 Deep Learning Model (Artificial Neural Network)

A **Multi-Layer Perceptron (MLP)** neural network was implemented using **TensorFlow / Keras**.

**ANN Architecture:**
Input Layer
↓
Dense (64 neurons, ReLU)
↓
Batch Normalization
↓
Dropout (0.2)
↓
Dense (32 neurons, ReLU)
↓
Batch Normalization
↓
Dropout (0.2)
↓
Sigmoid Output Layer

**Training Strategy:**

- Batch Normalization stabilizes training
- Dropout (20%) prevents overfitting
- Early Stopping based on validation loss

The ANN captures **complex nonlinear relationships** in transaction behavior.

---

### 4.3 Hybrid Ensemble Model (CatBoost + ANN)

To improve predictive performance, a **hybrid ensemble** was designed.

Both models generate **fraud probability predictions**:
CatBoost → P_catboost
ANN → P_ann


**Final Classification:** Threshold applied on `P_hybrid` for fraud detection.

**Why Hybrid Works:**  
CatBoost excels on structured data, while ANN captures nonlinear patterns. Combining both yields **stronger generalization and robustness**.

---

## 5. Experimental Evaluation

### 5.1 Evaluation Metrics

Performance assessed using:

- **Accuracy**
- **F1-Score**
- **ROC-AUC**

F1-Score is important due to **class imbalance**.

---

### 5.2 Synthetic Data-Based Robustness Testing

To ensure **behavioral learning**:

**Synthetic Fraud Profiles:**

- Short account lifetime
- Near-zero balance
- High transaction frequency
- Rapid fund transfers

**Synthetic Legitimate Profiles:**

- Long account lifetime
- High and stable balance
- Low transaction frequency

The model achieved **>99% confidence** on synthetic scenarios, confirming it learned **patterns rather than memorizing addresses**.

---

## 6. Results and Discussion

| Model                   | Accuracy  | F1-Score | ROC-AUC  | Remarks                                      |
|-------------------------|----------|----------|----------|----------------------------------------------|
| Hybrid (CatBoost + ANN) | 94.16%   | 0.9365   | 0.9864   | Best performance; most robust               |
| CatBoost Solo            | 93.97%   | 0.9334   | 0.9859   | Strong baseline                              |
| XGBoost Solo             | 94.01%   | 0.9339   | 0.9865   | High individual performance                  |
| LightGBM                 | 93.80%   | 0.9370   | 0.9370   | High efficiency                              |
| SVM                      | 92.84%   | N/A      | 0.9270   | Strong baseline                              |
| ANN Solo                 | 91.13%*  | 0.9399*  | 0.9846*  | (*Train metrics); captures nonlinear patterns |

**Overfitting Assessment:**  
Minimal performance gap between training and test:

- Train F1 (ANN): 93.99%  
- Test F1 (Hybrid): 93.25%  
- Delta < 1% → Confirms generalization

---

## 7. Implementation Details

### Technologies and Libraries

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning:** TensorFlow / Keras (Sequential API for ANN)
- **Visualization:** Matplotlib, Seaborn
- **Explainability:** SHAP
- **User Interface:** Streamlit

### ANN Architecture

- Input → Dense(64, ReLU) → BatchNorm → Dropout(0.2)  
- Dense(32, ReLU) → BatchNorm → Dropout(0.2)  
- Sigmoid Output Layer  
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Batch Size: 32  
- Early Stopping on validation loss

---

## 8. Reproducibility

### Installation

```bash
git clone https://github.com/Mouaad-Essa/Ethereum-Fraud-Detection.git
cd Ethereum-Fraud-Detection
pip install pandas numpy scikit-learn xgboost lightgbm catboost tensorflow matplotlib seaborn streamlit
```
### Usage

- Prepare Ethereum transaction data.

- Run preprocessing and feature engineering scripts.

- Train ML models and ANN.

- Evaluate hybrid ensemble predictions.

- Visualize results via Streamlit interface.
