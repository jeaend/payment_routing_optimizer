# 💳 **Payment Routing Optimizer**  

## 📝 **Overview**  
Develop a machine learning model to optimize the routing of credit card transactions to Payment Service Providers (PSPs). The goal is to minimize transaction fees, reduce failures, and maximize success rates by leveraging predictive modeling and cost optimization strategies.  

---

## 🎯 **Use Case**  
This project focuses on enhancing the efficiency of credit card transaction routing. The specific objectives include:  
- **Predictive Modeling:** Create a machine learning model to predict the likelihood of transaction success for each PSP in order to maximize transaction success and minimize costs.  

---

# 🔍 **Findings**  
This case study focuses on using machine learning to optimize credit card routing for online transactions, aiming to replace the current manual rule-based system with a more efficient, automated approach. The primary goal is to address issues with high payment failure rates, which lead to customer dissatisfaction and financial losses, by primarily increasing success rates. The CRISP-DM framework is used as a guideline throughout the project.

The final predictive model using XGBoost was selected for its robustness in handling imbalanced datasets and its ability to provide clear, actionable insights. Hyperparameter tuning was prioritized over resampling strategies, ensuring the model could effectively route transactions to the most appropriate Payment Service Provider (PSP). By predicting success probabilities for each PSP, the model supports dynamic routing decisions that balance cost and performance. The predictive model returns the probability of success for each PSP.

This solution can integrate into the existing infrastructure, delivering improvements in transaction success rates while fostering better customer experiences. The approach also offers deeper insights into payment behaviors and PSP performance.

---

## 📈 **Next Steps**  

1. **Dataset Limitations**  
   - The provided dataset was constructed and lacked meaningful patterns in the features.  
   - Features like the timestamp were not helpful in the current analysis.  
   - If there is an underlying issue with the data or feature representation, feature importance should be re-evaluated in future iterations.  

2. **Model Evaluation**  
   - Evaluate the model's impact on transaction success rates and overall fees using the final dataset.  
   - Fine-tune and determine an appropriate decision threshold to optimize precision and F1-score.  

3. **Iterative Training**  
   - Utilize the current approach as a starting point.  
   - Retrain the model with updated data as the imbalance improves over time, potentially leading to better performance and more balanced predictions.  

---

# 📚 **Documentation**  

## 🛠️ **Tools and Libraries**  
- **Programming Language:** Python.  
- **Libraries:**  
  - **EDA and Visualization:** `matplotlib`, `seaborn`.  
  - **Modeling:** `pandas`, `numpy`, `pickle`, `scikit-learn`, `xgboost`, `scipy`, `ibmlearn`.
    
---

## 🚀 **Getting Started**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/jeaend/payment_routing_optimizer.git
   cd payment_routing_optimizer
