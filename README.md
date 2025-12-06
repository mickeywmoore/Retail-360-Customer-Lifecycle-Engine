# 🛒 Retail-360: Intelligent Customer Lifecycle Engine

## 📋 Executive Summary
This project demonstrates an end-to-end Data Science pipeline designed to optimize customer retention and revenue. By analyzing raw transactional data, we aim to:
1.  **Engineer Features:** Transform raw SQL logs into a customer-level "RFM" (Recency, Frequency, Monetary) feature store.
2.  **Segment Customers:** Utilize Unsupervised Learning (K-Means) to identify distinct customer personas.
3.  **Predict LTV:** Build a Deep Neural Network (TensorFlow/Keras) to forecast future customer spending.
4.  **Audit for Bias:** Conduct an AI Ethics check to ensure equitable model performance.

---

# 📄 Model Card: Retail-360 LTV Engine

## Model Details
* **Developer:** Mickey Moore
* **Model Date:** December 2025
* **Model Version:** 1.0.0
* **Type:** Deep Neural Network (Regressor) for Customer Lifetime Value prediction.

## Intended Use
* **Primary Use Case:** Prioritizing marketing spend for high-value customers (Predicted LTV > $500).
* **Intended Users:** Marketing Analysts, CRM Managers.
* **Out of Scope:** Credit scoring or loan approval (model is not robust enough for financial risk assessment).

## Factors & Limitations
* **Data:** Trained on 10,000 synthetic transaction records (2017-2018).
* **Limitations:** The model degrades in accuracy for "New Customers" with less than 3 orders (Cold Start Problem).
* **Bias:** Audit performed on 'Region'. Max disparity in MAE was found to be < $5.00, deemed acceptable for marketing purposes.

## Ethical Considerations
* **Privacy:** All PII (Personally Identifiable Information) was hashed before training.
* **Fairness:** We explicitly optimized the model to minimize error variance across geographic regions to prevent "redlining" in marketing offers.