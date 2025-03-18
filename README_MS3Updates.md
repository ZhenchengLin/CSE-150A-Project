# Milestone3: Game Pricing Strategy using Bayesian Networks, Hidden Markov Models, and Reinforcement Learning

## PEAS/Agent Analysis

### Problem Background
The goal of this project is to develop an intelligent agent that predicts optimal pricing strategies for video games on the Steam platform. This pricing strategy should be based on various features such as developer, publisher, playtime, price history, and estimated owners. The agent aims to assist game developers in making pricing decisions that maximize revenue while considering customer engagement and retention.

### PEAS Framework
- **Performance Measure**: The agent's success is measured by its ability to accurately predict game prices that maximize estimated revenue while maintaining player engagement. We evaluate performance using actual historical data from the Steam dataset and compare revenue estimations.
- **Environment**: The agent operates in a dynamic digital marketplace where game pricing, user behavior, and competition influence the optimal strategy. The environment includes Steam games' historical sales data, playtime distributions, and market trends.
- **Actuators**: The agent suggests an optimal price given game attributes. In the case of reinforcement learning, it interacts iteratively with the environment by adjusting price policies.
- **Sensors**: The agent observes past game sales, playtime statistics, pricing, developer-publisher pairs, and owner estimations to infer optimal pricing strategies.

---

## Agent Setup, Data Preprocessing, and Training Setup

### Dataset Exploration and Key Variables
The dataset consists of Steam games' sales data, including attributes such as:
- **Developers & Publishers**: Categorical variables that influence game quality and market reach.
- **Release Date**: Temporal data affecting game popularity and pricing trends.
- **Median Playtime Forever**: A measure of user engagement, indicating how long players spend in the game.
- **Price**: A key decision variable in the model, affecting demand and revenue.
- **Estimated Owners**: Proxy for sales performance, dependent on price and user interest.
- **Revenue**: The final target metric, calculated as `Price * Estimated Owners`.

### Variable Interactions
The Bayesian Network model is structured as follows:
- **Developers, Publishers, and Release Date** → influence **Median Playtime Forever**.
- **Median Playtime Forever** → influences **Price**, as higher engagement often allows for higher pricing.
- **Price** → determines **Estimated Owners**, affecting the number of units sold.
- **Estimated Owners** → directly influences **Revenue**, which is the final measure of success.

Hidden Markov Models (HMMs) take a sequential approach by modeling the transitions between different pricing states over time. Reinforcement Learning (RL) is introduced to iteratively optimize pricing strategies based on observed revenue outcomes.

### Data Preprocessing
The preprocessing pipeline consists of:
1. **Feature Selection**: Extracting relevant attributes from the dataset.
2. **Data Cleaning**: Handling missing values and normalizing numerical features.
3. **Binning**: Categorizing numerical features into discrete bins for Bayesian Network modeling.
4. **Splitting**: Separating training and test sets to evaluate model performance.

### Calculating Parameters
For Bayesian Networks:
- Conditional Probability Tables (CPTs) are estimated from historical data.
- CPT formulas:
  - \( P(AveragePlaytime | Developers, Publishers) \)
  - \( P(Price | AveragePlaytime, MedianPlaytime) \)
  - \( P(EstimatedOwners | Price) \)

For HMMs:
- Transition probabilities between price states are learned from historical pricing sequences.
- The emission probabilities are estimated from game-specific features.

For RL:
- A reward function is defined as the revenue achieved after pricing decisions.
- Q-learning and SARSA algorithms are used to iteratively refine pricing strategies.

---

## Training the Model (10pts)
### Bayesian Network Implementation
```python
class BayesianNetworkModel:
    def Get_CPT_Avg_Median(self):
        parent_vars = ['Developers', 'Publishers']
        target_vars = ['Average playtime forever', 'Median playtime forever']
        cpt_tables = {}

        for target in target_vars:
            # Compute probability distribution for each category of target variable
            cpt = self.binidf.groupby(parent_vars)[target].value_counts(normalize = True).unstack().fillna(0)
            cpt_tables[target] = cpt  # Store CPT
        return cpt_tables
    def Get_CPT_Price(self):
        df_cpt_price = self.binidf.groupby(['Average playtime forever', 'Median playtime forever'])['Price'].value_counts(normalize=True).unstack().fillna(0)
        return df_cpt_price
    def Get_CPT_Estimated_Owners(self):
        df_cpt_estimated_owners = self.binidf.groupby(['Price'])['Estimated owners'].value_counts(normalize=True).unstack().fillna(0)
        return df_cpt_estimated_owners
    def Clean_Normalize(self, intidf, percent):
        # Define selected columns including AppID and set it as index
        # Define selected columns including environment factors (Developer, Publisher, Release Date)
        selected_columns = ['AppID', 'Name', 'Developers', 'Publishers', 'Release date' ,
                    'Average playtime forever', 'Median playtime forever', 'Estimated owners',
                    'Price', 'Positive', 'Negative']

        # Create a new DataFrame with only selected columns and set AppID as index
        df_selected = df[selected_columns].set_index('AppID')
        df_normalized = df_selected.copy()

        # Calculate total reviews (Positive + Negative)
        df_normalized['Total Reviews'] = df_normalized['Positive'] + df_normalized['Negative']

        # Avoid division by zero by replacing 0 total reviews with 1 (to keep ratio as 0)
        df_normalized['Total Reviews'] = df_normalized['Total Reviews'].replace(0, 1)

        # Normalize Positive and Negative reviews
        df_normalized['Positive Ratio'] = df_normalized['Positive'] / df_normalized['Total Reviews']
        df_normalized['Negative Ratio'] = df_normalized['Negative'] / df_normalized['Total Reviews']

        # Drop the 'Total Reviews' column as it's only needed for calculation
        df_normalized.drop(columns=['Total Reviews'], inplace=True)
        # Create a new DataFrame to store binned values
        df_binned = df_normalized.copy()

        # Define manual bin ranges for each numerical column with 3 levels (0,1,2)
        bin_ranges = {
            'Average playtime forever': [0, 5, 20000, float('inf')],  # Low: 0, Medium: 1-100, High: 100+
            'Median playtime forever': [0, 10, 40000, float('inf')],    # Low: 0, Medium: 1-50, High: 50+
            'Estimated owners': [0, 20000, 1000000, float('inf')],   # Low: 0-50K, Medium: 50K-500K, High: 500K+
            'Price': [0, 5, 40, float('inf')],                     # Low: 0-5, Medium: 5-20, High: 20+
            'Positive Ratio': [0, 0.5, 0.8, float('inf')],               # Low: 0-10, Medium: 10-1K, High: 1K+
            'Negative Ratio': [0, 0.5, 0.8, float('inf')]                         # Low: 0-40, Medium: 40-80, High: 80-100
        }
        # Apply manual binning using pd.cut() with 3 categories (0,1,2)
        for col, bins in bin_ranges.items():
            df_binned[col] = pd.cut(df_binned[col], bins=bins, labels=[0, 1, 2], include_lowest=True)
        df_binned['Average playtime forever'] = pd.to_numeric(df_binned['Average playtime forever'], errors='coerce')
        df_binned['Median playtime forever'] = pd.to_numeric(df_binned['Median playtime forever'], errors='coerce')
        df_shuffled = df_binned.sample(frac=1, random_state= np.random.randint(0, 10000) ).reset_index(drop=True)
        return df_shuffled.iloc[:int(percent*len(df_binned))], df_shuffled.iloc[int(percent*len(df_binned)):]
    def reinit(self, percent):
        self.binidf, self.testidf = self.Clean_Normalize(self.intidf, percent)
        self.cpt_avg_med = self.Get_CPT_Avg_Median()
        self.cpt_price = self.Get_CPT_Price()
        self.cpt_estimated_owners = self.Get_CPT_Estimated_Owners()
    def __init__(self, percent):
        """
        Initialize the Bayesian Network with given CPTs.

        :param cpt_dict: Dictionary of Conditional Probability Tables (CPTs)
        """
        self.intidf = pd.read_parquet("hf://datasets/FronkonGames/steam-games-dataset/data/train-00000-of-00001-e2ed184370a06932.parquet")
        self.binidf, self.testidf = self.Clean_Normalize(self.intidf, percent)
        self.cpt_avg_med = self.Get_CPT_Avg_Median()
        self.cpt_price = self.Get_CPT_Price()
        self.cpt_estimated_owners = self.Get_CPT_Estimated_Owners()
    def get_probabilityA(self, Developers,Publishers):
        """
        Compute the probability distribution of a target variable given conditions.

        :param target_var: The variable we want to predict.
        :param Publishers: str of given Publishers.
        :param Developers: str of given Developers.
        :return: What would be the most possible outcome of a new game from the given developers and publishers.
        """
        Probability_Avg = self.cpt_avg_med['Average playtime forever'].xs((Developers, Publishers), level=['Developers', 'Publishers']).values.tolist()[0]
        Probability_Median = self.cpt_avg_med['Median playtime forever'].xs((Developers, Publishers), level=['Developers', 'Publishers']).values.tolist()[0]
        Most_Possibe_Avg = np.argmax(Probability_Avg)
        Most_Possible_Median =  np.argmax(Probability_Median)
        Probability_price = self.cpt_price.loc[(Most_Possibe_Avg, Most_Possible_Median)].to_list()
        Most_Possible_Price = np.argmax(Probability_price)
        Probability_Estimated_Owners = self.cpt_estimated_owners.loc[Most_Possible_Price].to_list()
        Most_Possible_Estimated_Owners = np.argmax(Probability_Estimated_Owners)
        return Most_Possible_Price, Most_Possible_Estimated_Owners

    def get_range_description(self, column_name, bin_value):
        """
        Given a column name and a bin value (0,1,2), return the corresponding range description.

        :param column_name: The name of the feature (e.g., 'Price', 'Estimated owners').
        :param bin_value: The binned category (0, 1, or 2).
        :return: A string describing the value range.
        """
        bin_ranges = {
        'Average playtime forever': ["0 - 5", "5 - 20000", "20000+"],
        'Median playtime forever': ["0 - 10", "10 - 40000", "40000+"],
        'Estimated owners': ["0 - 20,000", "20,000 - 1,000,000", "1,000,000+"],
        'Price': ["$0 - $5", "$5 - $40", "$40+"],
        'Positive Ratio': ["0% - 50%", "50% - 80%", "80%+"],
        'Negative Ratio': ["0% - 50%", "50% - 80%", "80%+"]
          }

      # Ensure column exists in our range definitions
        if column_name not in bin_ranges:
            return "Invalid column name"

        # Ensure bin_value is valid
        if bin_value not in [0, 1, 2]:
            return "Invalid bin value"

        return bin_ranges[column_name][bin_value]

    # fixed verstion of get probability and this is use for testing much more cases
    def get_probability(self, Developers, Publishers):
      # Try to extract the probability distribution for "Average playtime forever"
      # for the given (Developers, Publishers) combination from the CPT.
      try:
          avg_series = self.cpt_avg_med['Average playtime forever'].xs(
              (Developers, Publishers), level=['Developers', 'Publishers']
          )
      except KeyError:
          # If the key (Developers, Publishers) is not found, return no prediction.
          return None, None

      # If the extracted series is empty, return no prediction.
      if avg_series.empty:
          return None, None

      # Convert the average playtime probabilities to a list.
      Probability_Avg = avg_series.values.tolist()
      # If the list is empty, return no prediction.
      if len(Probability_Avg) == 0:
          return None, None

      # Try to extract the probability distribution for "Median playtime forever"
      # for the given (Developers, Publishers) combination from the CPT.
      try:
          med_series = self.cpt_avg_med['Median playtime forever'].xs(
              (Developers, Publishers), level=['Developers', 'Publishers']
          )
      except KeyError:
          # If the key is not found, return no prediction.
          return None, None

      # If the extracted series is empty, return no prediction.
      if med_series.empty:
          return None, None

      # Convert the median playtime probabilities to a list.
      Probability_Median = med_series.values.tolist()
      # If the list is empty, return no prediction.
      if len(Probability_Median) == 0:
          return None, None

      # Select the bin (index) with the highest probability for average playtime.
      Most_Possible_Avg = np.argmax(Probability_Avg)
      # Select the bin (index) with the highest probability for median playtime.
      Most_Possible_Median = np.argmax(Probability_Median)

      # Use the most likely average and median bins to look up the Price CPT.
      try:
          price_probs = self.cpt_price.loc[(Most_Possible_Avg, Most_Possible_Median)].to_list()
      except KeyError:
          # If the Price CPT does not have data for these bins, return no prediction.
          return None, None

      # Choose the Price bin with the highest probability.
      Most_Possible_Price = np.argmax(price_probs)

      # Use the most likely Price bin to look up the Estimated Owners CPT.
      try:
          owners_probs = self.cpt_estimated_owners.loc[Most_Possible_Price].to_list()
      except KeyError:
          # If the Estimated Owners CPT does not have data for this Price bin, return no prediction.
          return None, None

      # Choose the Estimated Owners bin with the highest probability.
      Most_Possible_Estimated_Owners = np.argmax(owners_probs)

      # Return the predicted Price and Estimated Owners bins.
      return Most_Possible_Price, Most_Possible_Estimated_Owners


```
### HMM Implementation
```python
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

class GamePopularityHMM:
    def __init__(self, n_states=3, n_iter=200):
        """
        Initialize the Hidden Markov Model for predicting game popularity.

        :param n_states: Number of hidden states (e.g., 3 for Low, Medium, High popularity).
        :param n_iter: Number of iterations for model training.
        """
        self.n_states = n_states
        self.hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=n_iter)

    def preprocess_data(self, df):
        """
        Preprocess game data for HMM training.

        :param df: DataFrame containing game time-series features.
        :return: Processed feature matrix and lengths for HMM training.
        """
        feature_cols = [
            'Price', 'Median playtime forever', 'Positive', 'Negative',
            'Estimated owners', 'Peak CCU', 'Score rank'
        ]
        
        # Ensure all selected columns exist
        df = df[feature_cols].copy()

        # Convert all values to numeric, coercing errors (forces invalid strings to NaN)
        df = df.apply(pd.to_numeric, errors='coerce')

        # Fill missing values with column means to prevent NaN issues
        df.fillna(df.mean(), inplace=True)

        # Force any remaining NaNs to 0
        df.fillna(0, inplace=True)

        # Check for remaining NaNs
        if df.isnull().values.any():
            print("\n WARNING: NaN values detected after preprocessing!")
            print("NaN counts per column:\n", df.isnull().sum())
            print("\nForcing all NaNs to 0...\n")
            df.fillna(0, inplace=True)  # Forcefully replace any leftover NaNs

        # Normalize data (handling division-by-zero safely)
        df = (df - df.mean()) / (df.std() + 1e-8)

        # Convert DataFrame to NumPy array for HMM
        X = df.values
        lengths = [len(X)]

        return X, lengths


    def train(self, df):
        """
        Train the Hidden Markov Model on game data.

        :param df: DataFrame containing time-series data for training.
        """
        X, lengths = self.preprocess_data(df)
        self.hmm.fit(X, lengths)
        print("HMM training completed.")

    def predict_price_and_owners(self, df, developers, publishers):
        """
        Predict the most probable Price and Estimated Owners for a new game.

        :param df: DataFrame containing game time-series data.
        :param developers: Developer of the game.
        :param publishers: Publisher of the game.
        :return: Predicted price and estimated owners bins.
        """
        X, _ = self.preprocess_data(df)
        hidden_states = self.hmm.predict(X)

        # Find the most frequent hidden state associated with the given developer and publisher
        subset = df[(df['Developers'] == developers) & (df['Publishers'] == publishers)]
        if subset.empty:
            return None, None  # No prediction available

        most_common_state = np.argmax(np.bincount(hidden_states))

        # Map hidden states to most probable Price and Estimated Owners bins
        price_mapping = {0: 0, 1: 1, 2: 2}  # Adjust mapping based on observed trends
        owners_mapping = {0: 0, 1: 1, 2: 2}  # Same here, can be refined

        predicted_price_bin = price_mapping.get(most_common_state, 0)
        predicted_owners_bin = owners_mapping.get(most_common_state, 0)

        return predicted_price_bin, predicted_owners_bin

    def get_range_description(column_name, bin_value):
        """
        Given a column name and a bin value (0,1,2), return the corresponding range description.

        :param column_name: The name of the feature (e.g., 'Price', 'Estimated owners').
        :param bin_value: The binned category (0, 1, or 2).
        :return: A string describing the value range.
        """
        bin_ranges = {
            'Price': ["$0 - $5", "$5 - $40", "$40+"],
            'Estimated owners': ["0 - 20,000", "20,000 - 1,000,000", "1,000,000+"],
        }

        if column_name not in bin_ranges or bin_value not in [0, 1, 2]:
            return "Invalid value"

        return bin_ranges[column_name][bin_value]

```
### Reinforcement Learning Implementation
```python
# Additional analysis and visualizations
def analyze_results(results):
    """
    Perform additional analysis on the experiment results
    
    :param results: Results from run_game_pricing_optimization
    """
    # Extract components
    bayesian_model = results.get('bayesian_model')
    eval_results = results.get('eval_results', {})
    optimal_strategies = results.get('optimal_strategies')
    
    # Check if we have the necessary data
    if not eval_results or not eval_results.get('results'):
        print("Warning: Missing required data for analysis. Please run the full experiment first.")
        return {}
    
    # 1. Compare initial vs. optimal prices
    print("Initial vs. Optimal Prices Analysis:")
    if optimal_strategies is not None:
        plt.figure(figsize=(10, 6))
        
        developers = optimal_strategies['Developer']
        initial_prices = optimal_strategies['Initial Price']
        final_prices = optimal_strategies['Final Price']
        
        x = np.arange(len(developers))
        width = 0.35
        
        plt.bar(x - width/2, initial_prices, width, label='Initial Price')
        plt.bar(x + width/2, final_prices, width, label='Optimal Price')
        
        plt.xlabel('Developer')
        plt.ylabel('Price ($)')
        plt.title('Initial vs. Optimal Prices by Developer')
        plt.xticks(x, developers, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Calculate average price change
        avg_price_change = optimal_strategies['Price Change'].mean()
        print(f"Average price change across all developers: ${avg_price_change:.2f}")
        
        # Identify biggest price increases and decreases
        biggest_increase = optimal_strategies.loc[optimal_strategies['Price Change'].idxmax()]
        biggest_decrease = optimal_strategies.loc[optimal_strategies['Price Change'].idxmin()]
        
        print(f"Biggest price increase: {biggest_increase['Developer']} (${biggest_increase['Price Change']:.2f})")
        print(f"Biggest price decrease: {biggest_decrease['Developer']} (${biggest_decrease['Price Change']:.2f})")
    
    # 2. Analyze agent performance
    print("\nAgent Performance Analysis:")
    best_algo = eval_results.get('best_algorithm', 'Unknown')
    results_data = eval_results.get('results', {})
    
    # Check if results_data is a DataFrame or a dictionary
    if hasattr(results_data, 'items'):  # It's a dictionary
        # Compare final average rewards - make sure we're accessing the right structure
        try:
            final_rewards = {}
            for name, data in results_data.items():
                if isinstance(data, dict) and 'final_avg_reward' in data:
                    final_rewards[name] = data['final_avg_reward']
                
            if final_rewards:  # Only proceed if we have valid data
                algorithms = list(final_rewards.keys())
                reward_values = list(final_rewards.values())
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(algorithms, reward_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                plt.xlabel('RL Algorithm')
                plt.ylabel('Final Average Reward ($)')
                plt.title('Performance Comparison of RL Algorithms')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'${height:.2f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.show()
                
                print(f"Best algorithm: {best_algo} with final average reward: ${final_rewards.get(best_algo, 'N/A')}")
            else:
                print("Couldn't extract final average rewards from the results data.")
        except Exception as e:
            print(f"Error analyzing agent performance: {e}")
            print("Results data structure:", results_data)
    else:
        print("Results data is not in the expected format.")
    
    # 3. Analyze lifecycle pricing patterns
    if optimal_strategies is not None:
        try:
            print("\nLifecycle Pricing Analysis:")
            
            # Calculate average prices for each lifecycle stage
            avg_early_price = optimal_strategies['Optimal Early-Game Price'].mean()
            avg_mid_price = optimal_strategies['Optimal Mid-Game Price'].mean() 
            avg_late_price = optimal_strategies['Optimal Late-Game Price'].mean()
            
            lifecycle_data = {
                'Early Game': avg_early_price,
                'Mid Game': avg_mid_price, 
                'Late Game': avg_late_price
            }
            
            stages = list(lifecycle_data.keys())
            prices = list(lifecycle_data.values())
            
            plt.figure(figsize=(10, 6))
            plt.plot(stages, prices, marker='o', linewidth=2, markersize=10)
            plt.xlabel('Game Lifecycle Stage')
            plt.ylabel('Average Optimal Price ($)')
            plt.title('Average Optimal Price by Lifecycle Stage')
            
            # Add value labels
            for i, price in enumerate(prices):
                plt.text(i, price + 0.5, f'${price:.2f}', ha='center')
            
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Average Early-Game Price: ${avg_early_price:.2f}")
            print(f"Average Mid-Game Price: ${avg_mid_price:.2f}")
            print(f"Average Late-Game Price: ${avg_late_price:.2f}")
            
            # Determine typical pricing pattern
            if avg_early_price > avg_mid_price > avg_late_price:
                pattern = "Decreasing price over time (traditional pricing model)"
            elif avg_early_price < avg_mid_price > avg_late_price:
                pattern = "Increase then decrease (peak pricing model)"
            elif avg_early_price < avg_mid_price < avg_late_price:
                pattern = "Increasing price over time (value appreciation model)"
            else:
                pattern = "Mixed pricing pattern"
                
            print(f"Typical pricing pattern: {pattern}")
        except Exception as e:
            print(f"Error analyzing lifecycle pricing: {e}")
    
    # 4. Key insights summary
    print("\nKey Insights Summary:")
    print("---------------------")
    print(f"1. The {best_algo} algorithm was identified as the best performer for pricing optimization")
    
    if optimal_strategies is not None:
        try:
            # Calculate percentage of price increases vs decreases
            price_increases = (optimal_strategies['Price Change'] > 0).sum()
            price_decreases = (optimal_strategies['Price Change'] < 0).sum()
            price_unchanged = (optimal_strategies['Price Change'] == 0).sum()
            
            total_devs = len(optimal_strategies)
            pct_increase = (price_increases / total_devs) * 100
            pct_decrease = (price_decreases / total_devs) * 100
            pct_unchanged = (price_unchanged / total_devs) * 100
            
            if pct_increase > pct_decrease:
                print(f"2. The majority of developers ({pct_increase:.1f}%) would benefit from increasing their prices")
            elif pct_decrease > pct_increase:
                print(f"2. The majority of developers ({pct_decrease:.1f}%) would benefit from decreasing their prices")
            else:
                print(f"2. Developers are evenly split between price increases and decreases")
            
            if 'pattern' in locals():
                print(f"3. {pattern}")
        except Exception as e:
            print(f"Error generating key insights: {e}")
    
    return {
        'price_comparisons': {
            'initial_prices': initial_prices if 'initial_prices' in locals() else None,
            'final_prices': final_prices if 'final_prices' in locals() else None
        },
        'algorithm_performance': final_rewards if 'final_rewards' in locals() else None,
        'lifecycle_pricing': lifecycle_data if 'lifecycle_data' in locals() else None
    }
try:
    analysis = analyze_results(results)
except Exception as e:
    print(f"Analysis failed with error: {e}")
    import traceback
    traceback.print_exc()
```

---

## Conclusion and Results (20pts)

### Results and Visualizations
- **Bayesian Network**: Provided probabilistic estimates for pricing and ownership based on historical data. The CPTs allowed us to make informed predictions, but the model lacked adaptability to sudden market changes.
- **HMM**: Captured sequential trends in pricing strategies but struggled with unseen market shifts due to its dependence on historical transitions.
- **Reinforcement Learning**: RL outperformed other models in dynamic price optimization, achieving higher cumulative revenue by continuously adjusting prices.

#### Key Findings
- **Bayesian Networks**: Worked well for probabilistic inference but struggled with dynamically changing game trends.
- **Hidden Markov Models**: Effective for modeling price evolution but lacked adaptability to external factors.
- **Reinforcement Learning**: Adapted best to changing conditions, but required more training data and computational resources.

### Improvements
1. **Hybrid Models**: Combining Bayesian inference with RL for a more robust pricing strategy.
2. **Feature Engineering**: Introducing additional game metadata, such as user reviews, could refine predictions.
3. **Market Simulation**: Creating a synthetic pricing environment for training RL models more effectively.

This project demonstrates that while traditional probabilistic models provide interpretable insights, RL offers superior adaptability in dynamic pricing scenarios.

## 6. Acknowledgments and Contributions

- **Tianqi Li**: Data Cleaning, Original BN Design, HMM Design, README Writing  
- **Zhengcheng Lin**: BN Refinement, RL Design  
- **ChatGPT**: Debugging Assistance, Function Suggestions, Conclusion Writing  

### Libraries Used
- **pgmpy** for Bayesian Networks ([pgmpy.org](https://pgmpy.org/))  
- **hmmlearn** for Hidden Markov Models ([hmmlearn.readthedocs.io](https://hmmlearn.readthedocs.io/))  
- **gym** for Reinforcement Learning ([gymlibrary.dev](https://www.gymlibrary.dev/))  


