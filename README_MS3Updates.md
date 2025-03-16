# **CSE 150A Project - Milestone 3**

## **Overview**  

This project focuses on predicting **game popularity, pricing, and estimated owners** using probabilistic models. We implemented **Bayesian Networks (BN)** and **Hidden Markov Models (HMM)**, leveraging **Steam game data** to understand how **developers, publishers, pricing, and playtime trends** impact a game's success.

## **Major Updates**  

### **1. Hidden Markov Model (HMM) Implementation**
- Introduced **HMM-based prediction**, treating game popularity and pricing as latent states.
- Implemented **GaussianHMM** from `hmmlearn`, training the model using **Maximum Likelihood Estimation (MLE)**.
- Addressed **missing data** issues by applying **Iterative Imputation (EM Algorithm)**.
- **Results:** Performance was **lower than Bayesian Networks**, likely due to model assumptions and data sparsity.

### **2. Improved Bayesian Network (BN) Structure**
- **Refined BN structure** by improving conditional probability dependencies.
- Used **statistical analysis** to redefine **developer-publisher-playtime interactions**.
- Improved **binned probability distributions** for pricing and ownership estimations.
- **Results:** **BN consistently outperformed HMM**, especially in pricing and ownership predictions.

---

## **PEAS Analysis and Agent Setup**  

### **Problem Definition (PEAS Framework)**

| **Component** | **Description** |
|--------------|----------------|
| **Performance Measure** | Accuracy of predicted **pricing, playtime, and ownership trends**. |
| **Environment** | **Steam gaming industry**, where success depends on **developer reputation, pricing, and engagement metrics**. |
| **Actuators** | Game developers and publishers use model predictions to optimize **pricing and marketing strategies**. |
| **Sensors** | **Historical game data**: Playtime, pricing, developer-publisher metadata, user reviews, etc. |

### **Dataset & Features**  
- **Source:** [Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset)
- **Features Used:**
  - **Game Metadata:** Developer, Publisher, Release Date.
  - **Market Data:** Price, Owners, Revenue.
  - **Engagement Metrics:** Average & Median Playtime.
  - **User Sentiment:** Positive & Negative Reviews.

### **Preprocessing & Feature Engineering**  
- **Handling Missing Data:** Used **Iterative EM Imputation** for missing values.
- **Feature Normalization:** Applied **Min-Max Scaling** for numerical attributes.
- **Binning Strategies:** Converted **continuous price/playtime** into **discrete categories**.
- **Graphical Model Structure:** Designed BN & HMM **based on data dependencies**.

---

## **Model Training**

### **1. Bayesian Network (BN) Implementation**
```python
from BayesianNetwork import BayesianNetworkModel

# Initialize and train the BN model
bn_model = BayesianNetworkModel(percent=0.8)

# Predict price and owners for a new game by Valve
predicted_price, predicted_owners = bn_model.get_probability("Valve", "Valve")

# Get human-readable descriptions
price_range = bn_model.get_range_description("Price", predicted_price)
owners_range = bn_model.get_range_description("Estimated owners", predicted_owners)

print(f"Predicted Price: {price_range}")
print(f"Predicted Owners: {owners_range}")
```

### **2. Hidden Markov Model (HMM) Implementation**
```python
from HiddenMarkovModel import GamePopularityHMM

# Initialize and train the HMM model
hmm_model = GamePopularityHMM(n_states=3, n_iter=200)
hmm_model.train(df_games)

# Predict price and owners for a new game by Valve
predicted_price, predicted_owners = hmm_model.predict_price_and_owners(df_games, "Valve", "Valve")

# Get human-readable descriptions
price_range = GamePopularityHMM.get_range_description("Price", predicted_price)
owners_range = GamePopularityHMM.get_range_description("Estimated owners", predicted_owners)

print(f"Predicted Price: {price_range}")
print(f"Predicted Owners: {owners_range}")
```

---

## **Gaussian HMM Explanation**
The **Gaussian Hidden Markov Model (GaussianHMM)** is a continuous-state variant of the standard Hidden Markov Model (HMM). Unlike discrete HMMs, where each hidden state corresponds to a finite set of categorical outcomes, **GaussianHMM** assumes that emissions (observations) are drawn from a **multivariate Gaussian distribution**.

### **Key Differences from Discrete HMMs**
1. **Emission Probabilities:**  
   - In a discrete HMM, each state has a categorical probability distribution over a finite set of observations.  
   - In **GaussianHMM**, each state is associated with a **Gaussian distribution**, meaning it models **continuous-valued emissions**.
   
2. **Parameter Estimation:**  
   - The model is trained using **Expectation-Maximization (EM)** and **Baum-Welch algorithms**, estimating:
     - **Transition probabilities** between states.
     - **Means and covariances** for Gaussian emissions.
   
3. **Inference:**  
   - Uses **Viterbi Algorithm** to find the most probable sequence of states.
   - **Forward-Backward Algorithm** computes posterior probabilities.

### **Library Used:**
- `hmmlearn`: [Documentation](https://hmmlearn.readthedocs.io/en/latest/)  
  - Implements HMM models, including **GaussianHMM** for continuous emissions.

---

## **Evaluation Results & Insights**

### **Bayesian Network (BN) Results**
- **Price Prediction Accuracy:** 0.6019  
- **Owners Prediction Accuracy:** 0.7597  
- **Confusion Matrices Show Balanced Predictions Across Classes**  

### **Hidden Markov Model (HMM) Results**
- **Price Prediction Accuracy:** 0.5807  
- **Owners Prediction Accuracy:** 0.0452  
- **Confusion Matrices Show Skewed Predictions**  

### **Key Takeaways**
- **BN consistently outperforms HMM** due to its ability to leverage **direct conditional probabilities**.
- **HMM struggles with data sparsity**, leading to **poor owner predictions**.
- **Playtime-based transitions in HMM** do not effectively capture **non-sequential relationships** in the dataset.

---

## **Future Improvements**  

### **1. Improving HMM Training**
- **Use different initialization strategies** for emission and transition matrices.
- **Increase hidden states** to capture **more complex latent variables**.
- **Adjust feature selection** to remove redundant correlations.

### **2. Enhancing Data Representation**
- **Use embeddings** for categorical features (e.g., developer, publisher).
- **Incorporate additional metadata** like **game genres**.

### **3. Optimizing Bayesian Network**
- **Experiment with different BN structures** (using structure learning algorithms).
- **Increase granularity of bins** for **pricing and playtime distributions**.

---

## **Final Thoughts**  

The **Bayesian Network Agent** provides a **strong probabilistic framework** for predicting **game pricing and popularity**. The **HMM approach**, while valuable in sequential data modeling, **underperformed** due to **data sparsity** and **structural limitations**. **Future improvements** will focus on **optimizing HMM training, refining BN dependencies, and improving feature representations**.

### **Citations**
- **hmmlearn Library:** [https://hmmlearn.readthedocs.io/en/latest/](https://hmmlearn.readthedocs.io/en/latest/)
- **pgmpy Library for Bayesian Networks:** [https://pgmpy.org/](https://pgmpy.org/)
- **Dataset Source:** [https://huggingface.co/datasets/FronkonGames/steam-games-dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset)
