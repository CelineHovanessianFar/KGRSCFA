# 23: Adversarial Counterfactual Learning and Evaluation for Recommender System

Features:

Questions:

---

The authors introduced a counterfactual propensity-weighting approach within a minimax adversarial framework, where a candidate recommendation model (fθ) predicts user preferences, and an adversary model (gψ) characterizes the exposure mechanism. They used neural models like GMF, MLP, and attention-based networks to represent user and item features, applying propensity-weighted loss and adversarial training to mitigate exposure bias and improve recommendation accuracy.

---

## Genearal Summary

The paper discusses the challenges and limitations of traditional recommender systems, particularly focusing on the issues caused by feedback data being dependent on what products users were exposed to, but not accounting for the underlying exposure mechanisms. It introduces a novel counterfactual modeling framework based on causal inference to address these challenges.

Here's a breakdown of the key points covered in the paper:

1. **Problem Statement**: Recommender systems typically rely on user feedback (ratings, clicks) that are biased by the items users have been exposed to. Traditional models struggle because they do not consider how exposure to products influences user feedback.
2. **Counterfactual Propensity Weighting**: The authors highlight that not accounting for exposure mechanisms leads to biased results. They propose using counterfactual reasoning from causal inference to better model and understand user preferences under different exposure scenarios.
3. **Minimax Empirical Risk Formulation**: To deal with the partial observability of feedback data and the unobserved exposure mechanisms, the authors introduce a minimax risk strategy. This strategy involves an adversarial game between two models—a candidate recommendation model and a model that acts as an adversary by estimating the exposure mechanism.
4. **Adversarial Game**: The dual nature of the problem is converted into an adversarial game between the two models. The adversary model's role is to characterize the exposure mechanism, and this interaction is intended to minimize the worst-case scenario of exposure bias.
5. **Theoretical Contributions and Simulations**: The paper provides theoretical analyses demonstrating the inconsistencies in traditional supervised learning when applied to recommender systems without considering exposure. Through extensive simulations, the authors show that their approach can handle a wide range of recommendation settings and offers significant improvements over traditional methods.
6. **Empirical Evaluations**: They conduct experiments using synthetic and real data to validate their approach. The results suggest that their counterfactual and adversarial modeling significantly reduces bias and improves the accuracy of the recommender system.

Overall, this paper contributes to the field by addressing a critical gap in how recommender systems handle exposure bias, offering a robust framework that incorporates causal inference and adversarial learning to better predict user preferences under realistic scenarios.

---

## Technical Aspects

The technical solution proposed in the paper involves several key components that integrate ideas from causal inference and adversarial learning to address exposure bias in recommender systems:

1. **Counterfactual Propensity-Weighting**: This technique is borrowed from causal inference. It involves calculating the propensity score, which is the probability of an item being exposed to a user given the observed data. This score helps in reweighting the training data to reflect a more realistic scenario where exposure is independent of user choice, thus mitigating the bias due to observed data only.
2. **Minimax Empirical Risk Formulation**: The core of their technical solution is the formulation of the learning problem as a minimax optimization problem. In this framework, one model (the candidate recommendation model) tries to predict user preferences as accurately as possible, while the opposing model (characterizing the exposure mechanism) tries to maximize the first model's error by manipulating the understanding of the exposure mechanism. This adversarial setup is meant to ensure that the candidate model is robust against the worst-case scenarios of exposure bias.
3. **Adversarial Game Between Two Models**: The dual nature of the problem is expressed through an adversarial game. The adversary (second model) aims to characterize the underlying exposure mechanism by adjusting its parameters to maximize the prediction error of the primary recommendation model under the worst-case exposure scenario. This game leads to a scenario where both models iteratively update themselves to better predict or simulate the exposure, enhancing the robustness and accuracy of the overall system.
4. **Theoretical Analysis and Learning Bounds**: The paper provides a rigorous theoretical analysis to support their approach. They prove that traditional supervised learning applied to feedback data (without considering exposure mechanisms) can lead to inconsistent user preference detection. They also derive learning bounds for their adversarial learning setup, indicating how well the models can be expected to perform in general.
5. **Simulation and Empirical Validation**: To validate their approach, the authors conducted extensive simulation studies and empirical evaluations using both synthetic and real datasets. These studies demonstrate that their minimax adversarial model can significantly reduce bias introduced by exposure mechanisms and improve the accuracy and fairness of recommendations.

This combination of causal inference for exposure adjustment, adversarial learning for robustness, and theoretical underpinnings for consistency makes their approach technically rich and innovative for addressing exposure bias in recommender systems.

---

## Neural Model and Structural Details

Certainly! Here are the details about the neural models used in the paper, the representation of data, and additional technical aspects:

### **Neural Models Used**

1. **Recommendation Models (fθ and gψ)**:
    - **fθ (Candidate Model)**: This model predicts user preferences and is the primary recommendation model.
    - **gψ (Adversary Model)**: This model characterizes the underlying exposure mechanism and acts adversarially to challenge the candidate model.

The specific neural architectures for these models can vary, but common choices include:

- **Matrix Factorization (MF)**: A baseline collaborative filtering model where users and items are represented by latent factors.
- **Generalized Matrix Factorization (GMF)**: An extension of MF with element-wise product between user and item embeddings.
- **Multilayer Perceptron (MLP)**: A feedforward neural network with multiple layers to capture non-linear interactions between users and items.
- **Neural Collaborative Filtering (NCF)**: Combines GMF and MLP for better representation power.
- **Self-Attentive Sequential Model (Attn)**: Uses self-attention mechanisms to capture sequential dependencies in user interactions.

### **Data Representation**

1. **User and Item Features**:
    - **User Feature Vector (xu)**: Represents user-specific information. It could include user ID, demographic features, and past interaction history encoded as embeddings.
    - **Item Feature Vector (zi)**: Represents item-specific information. It could include item ID, category, textual descriptions, and other metadata encoded as embeddings.
2. **Exposure Status (Ou,i)**:
    - A binary variable indicating whether a user was exposed to an item (1 if exposed, 0 otherwise).
3. **Feedback (Yu,i)**:
    - Implicit Feedback: Binary variable indicating whether the user interacted with the item (e.g., click, purchase).
    - Explicit Feedback: Ratings given by the user to the item.

### **Technical Details**

1. **Propensity-Weighted Loss**:
    - The loss function is adjusted using propensity scores to reweight the observed feedback data. The propensity score *p*(*Ou*,*i*=1∣*xu*,*zi*) is used to ensure the learning process accounts for the non-random exposure.
        
        𝑝(𝑂𝑢,𝑖=1∣𝑥𝑢,𝑧𝑖)
        
2. **Minimax Formulation**:
    - The learning problem is formulated as a minimax game:
        
        ![Untitled](23%20Adversarial%20Counterfactual%20Learning%20and%20Evaluat%2014ee8c241f8a426c87bea19491782af4/Untitled.png)
        
    - 𝑊𝑐*Wc* denotes the Wasserstein distance, used to regularize the adversary model *gψ*.
        
        𝑔𝜓
        
3. **Adversarial Training**:
    - The models *fθ* and *gψ* are trained in an adversarial manner, where *fθ* aims to minimize the error, and *gψ* tries to maximize it by simulating the worst-case exposure mechanism.
        
        𝑓𝜃
        
        𝑔𝜓
        
        𝑓𝜃
        
        𝑔𝜓
        
4. **Learning Algorithm**:
    - They use a two-timescale gradient descent-ascent algorithm for training. *fθ* and *gψ* are updated iteratively with different learning rates to ensure stable training dynamics.
        
        𝑓𝜃
        
        𝑔𝜓
        
5. **Evaluation Metrics**:
    - **Hit Rate (Hit@k)**: Measures whether the true item is among the top-k recommendations.
    - **Normalized Discounted Cumulative Gain (NDCG@k)**: Measures the ranking quality of the recommendations, giving higher scores to hits at top positions.

### **Implementation Steps**

1. **Initial Setup**:
    - Initialize user and item embeddings.
    - Precompute propensity scores using an initial model.
2. **Model Training**:
    - **Stage 1**: Train the adversary model *gψ* to estimate the exposure mechanism.
        
        𝑔𝜓
        
    - **Stage 2**: Train the candidate model *fθ* using the propensity-weighted loss, with *gψ* providing the adversarial adjustments.
        
        𝑓𝜃
        
        𝑔𝜓
        
3. **Regularization**:
    - Use the Wasserstein distance as a regularizer to keep *gψ* close to the true exposure mechanism.
        
        𝑔𝜓
        
4. **Dual Problem Relaxation**:
    - Convert the primal minimax problem into a dual problem, making it more tractable for optimization.
5. **Online Evaluation**:
    - Validate the trained models using real-world user interaction data and compare the performance with online A/B tests.

In summary, the authors use a combination of advanced neural architectures for both candidate and adversary models, represent users and items with rich feature vectors, adjust for exposure bias using propensity-weighted losses, and optimize the models in an adversarial framework to ensure robust and accurate recommendations.

---

## Counterfactual Concept

The counterfactual concept is central to the authors' solution for addressing exposure bias in recommender systems. Here's how it is introduced and implemented:

1. **Counterfactual Modeling**: The authors employ counterfactual reasoning to answer "what if" questions regarding user feedback. Specifically, they ask what the feedback would have been if the user had been exposed to different items. This approach helps in understanding the causal relationships between exposure and user feedback.
1. **Propensity Scores**: To implement counterfactuals, the authors use propensity scores from causal inference. The propensity score is the probability of an item being exposed to a user, given the observed features. This score helps in reweighting the observed data to simulate a scenario where exposure is random, thus removing the bias introduced by the non-random exposure.
2. **Propensity-Weighted Loss**: They incorporate these propensity scores into a propensity-weighted loss function. This function adjusts the importance of each observed feedback instance based on the likelihood of the exposure, effectively simulating a counterfactual scenario where all items have an equal chance of being exposed.
3. **Minimax Adversarial Framework**: The counterfactual reasoning is further strengthened by embedding it within a minimax adversarial framework. The recommendation model (primary model) is trained to minimize the prediction error, while an adversary model is trained to maximize this error by simulating the worst-case exposure mechanism. This adversarial setup ensures that the recommendation model learns to be robust against potential biases in the exposure.
4. **Dual Problem and Adversarial Game**: They transform the minimax problem into a dual form, where the relaxation of the dual problem is expressed as an adversarial game between two recommendation models. The adversary (second model) characterizes the underlying exposure mechanism, and the primary model (candidate recommendation model) is optimized against this adversary. This setup implicitly accounts for counterfactual scenarios by making the primary model robust to different possible exposure mechanisms.
5. **Theoretical Guarantees**: The authors provide theoretical analysis to show that their approach leads to consistent estimates of user preferences by correctly accounting for the exposure mechanism. They derive generalization bounds for the learning process, ensuring that the model remains robust under various counterfactual scenarios.
6. **Empirical Validation**: Through simulations and real-world experiments, they demonstrate that models trained using their counterfactual and adversarial framework perform better in terms of reducing bias and improving recommendation accuracy. These experiments validate the practical utility of their counterfactual approach.

In summary, the counterfactual concept is introduced through the use of propensity scores to reweight observed data and is implemented within a robust adversarial framework that optimizes for worst-case exposure scenarios, ensuring that the recommendation model accurately captures user preferences under various hypothetical exposures.