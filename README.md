# CSE-150A-Project
Project repo for CSE 150A UCSD

[project doc link](https://docs.google.com/document/d/1n2NGoFP0WyeBzH93NpSXHTS5plMKLW4fhkT-MNVdAHw/edit?usp=sharing)

[dataset link](https://huggingface.co/datasets/FronkonGames/steam-games-dataset)






# Milestone 2 Update

---

# **Bayesian Network AI Agent for Predicting Game Popularity & Pricing**  

## **Overview**  
This project implements a **Bayesian Network AI Agent** to analyze and predict key aspects of a **new game's success** based on historical data. The model is trained on **Steam game data** and leverages **probabilistic inference** to predict:  

- **Game Popularity** (Average & Median Playtime)  
- **Optimal Pricing**  
- **Estimated Number of Owners**  

The AI agent follows a **probabilistic approach** using **Conditional Probability Tables (CPTs)** and **Bayesian Inference** to model relationships between **developers, publishers, pricing, playtime, and ownership trends**.  

---

## **Dataset**  
The dataset is sourced from [Hugging Face](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) and contains information on over **83,000+** Steam games, including:  
✅ **Game Metadata** – Developer, Publisher 
✅ **Market Data** – Price, Owners, Revenue  
✅ **Engagement Metrics** – Average & Median Playtime
✅ **User Reviews** – Positive and Negative Ratings  

---

## **Modeling Approach**  

### **1️⃣ Bayesian Network Structure**
The model is structured as a **directed acyclic graph (DAG)**:  

1. **Developers & Publishers** → Predict **Playtime (Popularity)**  
2. **Playtime (Popularity)** → Predict **Price**  
3. **Price** → Predict **Estimated Owners (Sales)**  
4. **Estimated Owners + Price** → Predict **Revenue**  

### **2️⃣ Conditional Probability Tables (CPTs)**  
We compute CPTs for key variables:  

| Parent Variables | Target Variable | Function |
|-----------------|-----------------|----------|
| Developers, Publishers | Average Playtime Forever, Median Playtime Forever | `Get_CPT_Avg_Median()` |
| Average Playtime, Median Playtime | Price | `Get_CPT_Price()` |
| Price | Estimated Owners | `Get_CPT_Estimated_Owners()` |

---

## **Implementation Details**

### **🔹 Data Preprocessing & Normalization**  
To simplify numerical features, we manually **bin** continuous values into **three categories (0,1,2):**  

| Feature | Bin 0 (Low) | Bin 1 (Medium) | Bin 2 (High) |
|---------|------------|---------------|-------------|
| **Average Playtime Forever** | 0 - 5 mins | 5 - 20,000 mins | 20,000+ mins |
| **Median Playtime Forever** | 0 - 10 mins | 10 - 40,000 mins | 40,000+ mins |
| **Estimated Owners** | 0 - 20,000 | 20,000 - 1,000,000 | 1,000,000+ |
| **Price** | $0 - $5 | $5 - $40 | $40+ |
| **Positive Ratio** | 0% - 50% | 50% - 80% | 80%+ |
| **Negative Ratio** | 0% - 50% | 50% - 80% | 80%+ |

**Function:** `Clean_Normalize()`  

---

### **🔹 AI Agent with Bayesian Inference**  
The AI agent performs **probabilistic queries** to predict the **most likely outcome** for a new game.  

**Example:**  
💡 *Predicting the price of the next game by Valve*  
```python
# Initialize the Bayesian Model
bayesian_model = BayesianNetworkModel(percent=0.8)

# Predict the most probable price category for the next Valve game
developer_name = "Valve"
publisher_name = "Valve"

predicted_price = bayesian_model.get_probability(developer_name, publisher_name)
print(f"Predicted price category: {predicted_price}")
```
➡️ **Output:** `"Most probable price category: 1 (Medium, $5 - $40)"`

---

## **Functions Implemented**

### **1️⃣ Conditional Probability Estimation**
```python
def Get_CPT_Avg_Median(self):
def Get_CPT_Price(self):
def Get_CPT_Estimated_Owners(self):
```
📌 Computes **normalized probability tables** based on past data.

---

### **2️⃣ Querying the Model**
```python
def get_probability(self, Developers, Publishers):
```
📌 Returns **most probable playtime, price, and estimated owners** for a new game.

---

### **3️⃣ Interpreting Binned Values**
```python
def get_range_description(column_name, bin_value):
```
📌 Converts **0,1,2** bins into **human-readable ranges.**

Example:
```python
print(get_range_description("Price", 2))  # Output: "$40+"
```

---

### **4️⃣ Finding the Most Likely Category**
```python
def get_most_probable_category(distribution):
```
📌 Given a **list of probabilities**, returns the **most likely outcome**.

Example:
```python
prob_list = [0.8, 0.1, 0.1]
print(get_most_probable_category(prob_list))  # Output: 0 (Low)
```

---

## **Evaluation & Next Steps**

### ✅ **Current Achievements**
🔹 **Fully implemented Bayesian Network**  
🔹 **CPTs computed & stored efficiently**  
🔹 **Model can predict price, playtime, and ownership trends**  
🔹 **Interpretable binning strategy applied to numerical values**  

### 🔜 **Future Improvements**
🔹 **Optimize inference speed** (Currently queries are direct CPT lookups)  
🔹 **Add online learning** (Update CPTs dynamically with new games)  
🔹 **Incorporate deep learning** (Use Bayesian Neural Networks)  

---

## **How to Run**
### **🔹 Step 1: Install Requirements**
```bash
pip install pandas numpy
```

### **🔹 Step 2: Run the Model**
```python
from bayesian_model import BayesianNetworkModel

# Initialize model
model = BayesianNetworkModel(percent=0.8)

# Predict next Valve game price
print(model.get_probability("Valve", "Valve"))
```

---

## Conclusion

The Bayesian Network AI Agent offers a robust and interpretable probabilistic framework for predicting key aspects of a new game's success, including playtime metrics, pricing strategy, and estimated ownership. By leveraging historical data from over 83,000 Steam games, the model constructs Conditional Probability Tables (CPTs) to capture the nuanced relationships between developers, publishers, playtime, price, and ownership.

Key achievements of the project include:

- **Probabilistic Modeling:** A clear, step-by-step Bayesian inference process that starts with developers and publishers to predict playtime, which in turn informs pricing and ultimately the estimated number of owners.
- **Data Simplification:** A manual binning strategy that transforms continuous variables into three meaningful categories, enhancing interpretability.
- **Efficient Inference:** Direct CPT lookups ensure that the model provides quick, interpretable predictions for new games.
- **Actionable Insights:** The agent aids game developers and analysts in making data-driven decisions regarding pricing and marketing strategies.

Looking ahead, potential improvements include:

- **Optimization of Inference Speed:** Further streamlining the CPT lookup process to handle larger datasets efficiently.
- **Dynamic Online Learning:** Updating the CPTs in real-time as new game data becomes available.

Overall, this Bayesian Network AI Agent serves as a strong foundation for predictive analytics in the gaming industry, demonstrating both practical applicability and significant room for future enhancement.
