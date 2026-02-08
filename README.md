# Ethereum Fraud Detection System

## A Machine Learning-Based Approach for Detecting Illicit Activities on the Ethereum Blockchain

**Academic Project (Master 2 – PFM, Mouaad Es-safryouy | Ilyass Bouchtaoui)**

---

## Abstract

Blockchain systems such as Ethereum provide transparency and decentralization, but their pseudo-anonymous nature also facilitates illicit activities including money laundering, phishing attacks, and Ponzi schemes. This project proposes a machine learning–based fraud detection system that analyzes Ethereum transaction behavior to classify wallet addresses as either legitimate or fraudulent.

The proposed solution relies on extensive feature engineering, benchmarking of multiple state-of-the-art classification algorithms, and the implementation of a two-level stacking ensemble architecture. To ensure that the model learns meaningful behavioral patterns rather than memorizing historical data, robustness was evaluated using synthetically generated transaction scenarios. Experimental results demonstrate that gradient boosting models, particularly XGBoost, achieve strong predictive performance, while the stacking ensemble provides improved robustness and generalization.

---

## 1. Introduction

The rapid growth of blockchain-based financial systems has introduced new challenges in transaction monitoring and fraud detection. Unlike traditional banking systems, Ethereum does not rely on centralized identity verification, which complicates the detection of illicit behaviors at the address level.

Machine learning techniques offer a promising solution by modeling transaction dynamics and behavioral patterns over time. However, a major challenge lies in ensuring that learned models generalize to unseen fraud strategies rather than overfitting to known addresses. This project addresses this challenge by combining ensemble learning with synthetic data validation to assess the logical consistency of model predictions.

---

## 2. Objectives

The primary objectives of this project are as follows:

- To design a complete machine learning pipeline for Ethereum fraud detection.
- To extract and engineer meaningful financial and temporal features from raw blockchain transaction data.
- To benchmark classical and state-of-the-art machine learning models on the fraud detection task.
- To improve predictive performance and robustness using a stacking ensemble strategy.
- To validate model generalization using artificially generated transaction behaviors.

---

## 3. Data Processing and Feature Engineering

Raw Ethereum transaction logs were processed to construct a structured dataset at the wallet-address level. Feature engineering focused on capturing both **financial characteristics** and **temporal dynamics**, including:

- Transaction frequency and inter-arrival times  
- Total incoming and outgoing transaction values  
- Account lifetime and activity duration  
- Balance-related statistics and variability measures  

These features aim to reflect behavioral patterns commonly associated with fraudulent and legitimate users.

---

## 4. Methodology

### 4.1 Base Models (Level 0)

A diverse set of classifiers was selected to capture complementary perspectives on the data:

- **XGBoost**  
  Gradient boosting decision trees optimized for tabular data. This model achieved the highest individual performance.

- **LightGBM**  
  A fast and memory-efficient gradient boosting framework suitable for large-scale data.

- **CatBoost**  
  Included for its stability and reduced sensitivity to feature preprocessing.

- **ExtraTrees Classifier**  
  An Extremely Randomized Trees model introduced to increase ensemble diversity and reduce variance through additional randomization.

- **Random Forest and Support Vector Machine (SVM)**  
  Used as traditional baselines.

- **Multi-Layer Perceptron (MLP)**  
  A fully connected neural network used to assess the effectiveness of deep learning on tabular blockchain data.

- **TabNet**  
  An attention-based deep learning architecture designed specifically for tabular datasets.

---

### 4.2 Stacking Ensemble (Level 1)

To improve robustness and generalization, a two-level stacking ensemble was implemented:

- Predictions from all base models constitute the input feature space of the meta-level.
- A **Logistic Regression** classifier serves as the meta-learner.
- The meta-model learns how to optimally combine the strengths of individual classifiers.

This approach reduces reliance on a single model and mitigates overfitting.

---

## 5. Experimental Evaluation

### 5.1 Evaluation Metrics

Model performance was assessed using standard classification metrics:

- Accuracy  
- Receiver Operating Characteristic Area Under the Curve (ROC-AUC)  
- F1-Score  

These metrics provide a balanced evaluation in the presence of class imbalance.

---

### 5.2 Synthetic Data-Based Robustness Testing

To verify that the model learned meaningful fraud-related behavior rather than memorizing training samples, a synthetic data validation strategy was adopted.

#### Synthetic Scenario Generation

Artificial Ethereum wallet profiles were generated to simulate extreme yet realistic behaviors:

- **Fraudulent Profiles**  
  - Very short account lifetime  
  - Near-zero balance  
  - High transaction frequency  
  - Rapid fund movement  

- **Legitimate Profiles**  
  - Long account lifetime  
  - High and stable balances  
  - Low transaction frequency  

#### Validation Procedure and Results

The trained **XGBoost model** was evaluated on this unseen synthetic dataset. The model achieved classification confidence exceeding **99%** on the generated scenarios, indicating that it correctly learned behavioral fraud patterns rather than relying on address memorization.

---

## 6. Results and Discussion

| Model        | Accuracy | ROC-AUC | Remarks |
|-------------|----------|---------|--------|
| XGBoost     | 94.01%   | 0.939   | Best individual model |
| Stacking    | 93.92%   | 0.938   | Most robust overall |
| LightGBM    | 93.80%   | 0.937   | High efficiency |
| CatBoost    | ~93.5%   | 0.935   | Stable performance |
| SVM         | 92.84%   | 0.927   | Strong baseline |
| MLP         | 87.50%   | 0.872   | Limited effectiveness |

While XGBoost achieved the highest individual accuracy, the stacking ensemble demonstrated superior robustness across evaluation settings.

---

## 7. Implementation Details

### Technologies and Libraries

- **Programming Language**: Python  
- **Data Processing**: Pandas, NumPy  
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost  
- **Deep Learning**: TensorFlow/Keras, PyTorch, PyTorch-TabNet  
- **Visualization and Explainability**: Matplotlib, Seaborn, SHAP  
- **User Interface**: Streamlit  

---

## 8. Reproducibility

### Installation

```bash
git clone https://github.com/Mouaad-Essa/Ethereum-Fraud-Detection.git
cd Ethereum-Fraud-Detection
pip install pandas numpy scikit-learn xgboost lightgbm catboost tensorflow torch pytorch-tabnet matplotlib seaborn streamlit
