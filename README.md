# 💳 **Payment Routing Optimizer**  

## 📝 **Overview**  
Develop a machine learning model to optimize the routing of credit card transactions to Payment Service Providers (PSPs). The goal is to minimize transaction fees, reduce failures, and maximize success rates by leveraging predictive modeling and cost optimization strategies.  

---

## 🎯 **Use Case**  
This project focuses on enhancing the efficiency of credit card transaction routing. The specific objectives include:  
- **Predictive Modeling:** Create a machine learning model to predict the likelihood of transaction success for each PSP in order to maximize transaction success and minimize costs.  

---


# 🔍 **Findings**  
- **TO DO:**

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
