# 9: Counterfactual Explanations for Neural Recommenders

Features:

- -

---

The authors developed ACCENT, a framework for generating counterfactual explanations in neural recommender systems by extending influence functions to pairs of items. They adapted the Fast Influence Analysis (FIA) method to efficiently estimate how changes in user-item interactions affect recommendations. This method was validated on two neural models, Neural Collaborative Filtering (NCF) and Relational Collaborative Filtering (RCF), using the MovieLens 100K dataset, demonstrating its ability to produce minimal and meaningful changes that lead to alternative recommendations.

---

This paper introduces ACCENT, a framework for generating counterfactual explanations in neural recommender systems. Unlike previous methods that heavily relied on attention mechanisms or other complex, sometimes unactionable features, ACCENT focuses on a user's specific interactions to provide tangible and scrutable explanations. It extends the idea of influence functions—originally designed to evaluate the impact of a data point on model training—to handle pairs of items, identifying training points most relevant to a recommendation. This allows ACCENT to iteratively deduce a counterfactual set: a minimal set of the user's actions that, if removed, would lead to a different recommendation.

The framework is applied to two types of neural recommender models, Neural Collaborative Filtering (NCF) and Relational Collaborative Filtering (RCF), and is tested on the MovieLens 100K dataset. By adjusting influence functions for deep learning, ACCENT can manage the complexity of neural networks, effectively bridging the gap between traditional, simpler models and more advanced, non-linear systems. The results demonstrate ACCENT's feasibility and effectiveness, providing statistically significant improvements in creating smaller and more accurate counterfactual explanations compared to baseline methods.

---

## Technical Aspects

The technical solution proposed in the paper revolves around the application of influence functions to provide counterfactual explanations in neural recommender systems. Here's a breakdown of the key technical components and concepts used in the ACCENT framework:

### **1. Influence Functions**

Influence functions are a statistical tool used to measure how much the change in a single data point's weight affects the overall model parameters. Traditionally, these are used to understand how a small perturbation in the training data impacts the outcome predicted by the model.

### **2. Extension to Neural Recommenders**

ACCENT extends the use of influence functions from single data points to pairs of items. This is crucial because, in neural recommenders, the relationships and interactions between items (like products or movies) significantly influence recommendations.

### **3. Counterfactual Explanation Generation**

The process of generating counterfactual explanations involves the following steps:

- **Identifying Influence**: ACCENT first calculates how individual user actions (like viewing or rating a movie) influence the recommendation. This is done by estimating how changes in the model parameters affect the predicted recommendation score.
- **Pairwise Influence**: The influence on the recommendation score difference between two items (a recommended item and its potential replacement) is computed. This helps in understanding how swapping one item with another affects the recommendation.
- **Iterative Process**: ACCENT uses this pairwise influence to iteratively find a set of user actions that, when hypothetically removed, would lead to a different recommendation. This set of actions forms the counterfactual explanation.

### **4. Fast Influence Analysis (FIA)**

FIA is an adapted method used to compute the influence of removing a data point efficiently, especially in large-scale neural models where computational cost is a concern. ACCENT incorporates FIA to reduce computational demands while maintaining the accuracy of influence estimates.

### **5. Practical Implementation**

For implementation, ACCENT modifies the typical influence function calculations to fit the neural network context:

- **Hessian Computation**: The Hessian matrix, which contains second-order derivatives of the loss function with respect to the model parameters, is crucial for calculating influence. However, since neural models can be non-convex and large, ACCENT adds a damping term to ensure that the matrix is invertible.
- **Influence Estimation**: The influence of removing a user action is calculated by estimating how the removal affects the model parameters, and thus the predicted scores for recommendations.

### **6. Evaluation and Validation**

ACCENT was validated on the MovieLens 100K dataset using two neural models, NCF and RCF. The framework was compared against baseline methods like attention-based and other influence-based approaches. The evaluation focused on the effectiveness of counterfactual explanations in terms of their size (number of actions in the explanation set) and their impact on recommendation changes.

Overall, ACCENT provides a robust framework for generating actionable and understandable counterfactual explanations for complex neural recommender systems by leveraging the computational techniques adapted from influence functions and applying them to the specific needs of recommendation models.

---

## Neural Model and Structural Details

the authors use two specific types of neural recommender models: Neural Collaborative Filtering (NCF) and Relational Collaborative Filtering (RCF). Each of these models approaches the recommendation problem with different architectures and data representations:

### **1. Neural Collaborative Filtering (NCF)**

**Model Architecture:**

- NCF combines a generalized matrix factorization (GMF) model with a multilayer perceptron (MLP) to learn the non-linear and complex interactions between users and items. This hybrid approach captures both linear interactions (through GMF) and non-linear patterns (through MLP).
- The final layer of the model fuses features learned by both GMF and MLP paths to predict the final recommendation score, which is used to determine the most suitable items for a user.

**Data Representation:**

- NCF typically uses user-item interaction data, which is often binary (e.g., click or no click, purchase or no purchase).
- In this study, the input data for NCF is derived from the MovieLens 100K dataset. Ratings are likely binarized, where ratings above a certain threshold (e.g., 3 out of 5) are treated as positive interactions and the rest as negative.

### **2. Relational Collaborative Filtering (RCF)**

**Model Architecture:**

- RCF extends the idea of collaborative filtering by incorporating auxiliary information about item-item relations, which helps in learning better item embeddings. This model uses a two-layer attention mechanism to compute target-aware embeddings that reflect not only user-item interactions but also item-item relationships.
- These embeddings are then used to predict the recommendation score, taking into account both direct user-item interactions and the context provided by related items.

**Data Representation:**

- RCF uses a richer dataset that includes not only user-item interactions but also item-item relationships. These relationships could be based on similarities between items, such as genre or brand associations.
- For the MovieLens dataset, this might include metadata about the movies that relate them to each other, like similar genres, directors, or actors.

### **Additional Technical Details:**

**Counterfactual Explanation Methodology:**

- Both models incorporate a methodology where they calculate influence scores to determine how specific user actions affect the outcome. These influence scores are derived using Fast Influence Analysis (FIA), adapted for the complexity of neural networks.
- ACCENT, the framework developed in the paper, iteratively computes the influence of removing pairs of items (or changing interactions) to generate a minimal set of user actions that, if altered, would lead to a different recommendation. This set is presented as a counterfactual explanation to the user.

**Dataset Handling and Preparation:**

- For RCF, the ratings in the MovieLens dataset are binarized to fit the implicit feedback model, where interactions are either positive (ratings ≥ 3) or negative (ratings < 3).
- Users with less than a certain number of positive or negative ratings are pruned to ensure that the remaining user profiles are sufficiently informative for the learning algorithms.

**Experimental Setup:**

- The framework was tested using a subset of the MovieLens 100K dataset, ensuring a balanced and representative sample of user interactions.
- Evaluations compared the effectiveness of ACCENT against several baseline algorithms, using metrics such as the size of the counterfactual explanation set and the percentage of recommendations that changed as intended when the counterfactual actions were hypothetically removed.

These details highlight the sophisticated nature of the models used and the comprehensive approach taken to develop and validate a method for generating tangible and actionable explanations in neural recommender systems.