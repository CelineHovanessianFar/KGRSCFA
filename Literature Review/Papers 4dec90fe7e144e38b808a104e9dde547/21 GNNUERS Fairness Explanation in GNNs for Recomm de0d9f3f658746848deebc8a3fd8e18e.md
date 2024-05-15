# 21: GNNUERS: Fairness Explanation in GNNs for Recommendation via Counterfactual Reasoning

Features:

Questions:

---

The researchers developed GNNUERS, a framework using Graph Neural Networks (GNNs) to understand and mitigate unfairness in recommender systems. They represented user-item interactions as a bipartite graph and applied counterfactual reasoning by perturbing this graph—specifically, by selectively removing edges (user-item interactions) to simulate scenarios that could reduce unfairness. The framework evaluates changes using a loss function that balances demographic parity and the minimality of graph alterations, aiming to systematically explain and address disparities in recommendation utility across different demographic groups.

---

## Genearal Summary

This paper presents a novel algorithm, GNNUERS, which aims to identify and explain user unfairness in graph neural network (GNN)-based recommender systems by focusing on user-item interactions in a bipartite graph. The approach involves perturbing the bipartite graph's structure to find a modified version that reduces utility disparity between protected and unprotected demographic groups. This perturbation aims to provide explanations for observed unfairness by highlighting specific user-item interactions that contribute to bias.

Key contributions of this work include:

1. A counterfactual reasoning framework that modifies user-item interactions to minimize demographic disparities in utility, offering a new way to address fairness in recommendations.
2. Extensive experiments across several real-world datasets and state-of-the-art GNN models, demonstrating the method's ability to systematically explain unfairness.
3. The use of a bipartite graph structure specific to recommender systems, which is different from typical graphs used in other GNN applications, highlighting unique challenges like efficiency issues due to the graph's bipartite nature.

This research advances our understanding of algorithmic fairness in recommender systems by showing how altering specific interactions can influence the overall fairness of the system, thereby supporting system designers and service providers with actionable insights on improving their models.

---

## Technical Aspects

The technical solution presented in the paper involves several key components that together form the GNNUERS framework, which stands for Graph Neural Network-based Unfairness Explainer in Recommender Systems. Here’s a breakdown of the main technical aspects:

1. **Bipartite Graph Representation**:
    - The user-item interactions in a recommender system are represented as a bipartite graph, where users and items form two distinct sets of nodes, and edges between these nodes represent interactions like ratings or purchases.
    - This representation is essential because it reflects the real structure of interactions in a recommender system, allowing the model to focus on the links that directly influence recommendations.
2. **Counterfactual Reasoning**:
    - The core of GNNUERS involves counterfactual reasoning, which in this context means modifying the graph's structure to explore how changes would affect the output of the recommender system.
    - Specifically, it involves perturbing (adding or removing) edges in the bipartite graph to simulate "what-if" scenarios. These scenarios help identify which interactions, if altered, would lead to a reduction in the observed unfairness.
3. **Perturbation Mechanism**:
    
    
    - The algorithm selects specific user-item interactions (edges) to perturb. This selection is driven by a loss function that combines two terms: minimizing the disparity in recommendation utility across demographic groups and minimizing the number of perturbed edges.
    - The objective is to find a subset of edges whose removal or alteration would most effectively reduce the unfairness while impacting the smallest number of other interactions.
4. **Demographic Parity**:
    - The framework aims to achieve demographic parity, a fairness criterion ensuring that all demographic groups receive recommendations of roughly equal utility.
    - The perturbation is guided by this principle, aiming to adjust the graph structure in such a way that the utility of recommendations becomes more evenly distributed across different groups.
5. **Loss Function**:
    - The loss function used in GNNUERS is crucial for guiding the perturbation process. It measures the difference in utility between protected and unprotected groups and aims to minimize this difference.
    - The function includes terms for both the fairness objective (demographic parity) and a regularization term that controls the magnitude of the perturbation, helping to ensure that the original graph's structure is not excessively distorted.
6. **Empirical Evaluation**:
    - The approach is empirically evaluated on multiple real-world datasets using state-of-the-art GNN models. This evaluation helps validate the effectiveness of the proposed modifications in reducing unfairness.
    - The evaluation considers the impact of perturbations on recommendation accuracy and fairness, providing a balanced view of how the changes affect overall system performance.

Overall, GNNUERS introduces a novel approach to addressing unfairness in recommender systems by leveraging the structural properties of bipartite graphs and applying counterfactual reasoning to uncover and explain the sources of bias. This method provides both a theoretical and practical framework for improving fairness in recommendations.

---

## Neural Model and Structural Details

The GNNUERS framework, as discussed in the paper, leverages Graph Neural Networks (GNNs) to model the interactions between users and items within a recommender system. Here’s a detailed look at the neural models used and the data representation approach, along with other pertinent technical details:

### **Neural Model Used**

1. **Graph Neural Networks (GNNs)**:
    - GNNUERS utilizes state-of-the-art GNN models that are specifically designed for handling graph-structured data. These models are capable of capturing the complex dependencies within the bipartite graph of user-item interactions, which is crucial for accurately modeling and subsequently altering the recommendation process to enhance fairness.
2. **Specific GNN Architectures**:
    - The paper mentions the use of several GNN-based recommender systems for evaluation purposes, including popular models like GCMC (Graph Convolutional Matrix Completion), NGCF (Neural Graph Collaborative Filtering), and LightGCN. Each of these models has distinct characteristics:
        - **GCMC**: Utilizes graph convolution operations to encode user-item interactions into embeddings, which are then used to predict missing interactions.
        - **NGCF**: Emphasizes capturing higher-order connectivities by propagating user and item embeddings through the graph structure, thereby enriching the embeddings with neighborhood information.
        - **LightGCN**: Simplifies the design of GCN for recommendation by removing feature transformation and non-linear activation, focusing solely on the essential neighborhood aggregation to enhance recommendation performance.

### **Data Representation**

1. **Bipartite Graph Structure**:
    - The user-item interactions are represented as a bipartite graph where one set of nodes represents users and the other set represents items. Edges between these nodes represent interactions, such as ratings or purchases, and are the primary focus for perturbation in the counterfactual reasoning process.
2. **Feature Representation**:
    - While the paper primarily discusses structural modifications (i.e., which edges to add or remove), the underlying feature representation of users and items (if used beyond IDs) typically involves embedding vectors that capture the characteristics or preferences implicit in their interactions. These embeddings are learned during the training of the GNN models and are crucial for both predicting interactions and understanding the impact of perturbations.

### **Implementation Details**

1. **Loss Function**:
    
    
    - The counterfactual perturbations are guided by a loss function designed to balance fairness (reducing disparity across groups) and sparsity (minimizing the number of perturbations). This function typically includes terms for demographic parity (ensuring equal utility distribution across groups) and regularization to maintain the graph's integrity.
2. **Optimization Process**:
    - The GNNUERS framework employs an optimization process where the graph's structure is iteratively adjusted based on the loss function. This involves using gradient-based methods to find the optimal set of edges to perturb that would lead to the desired improvements in fairness.
3. **Evaluation Metrics**:
    - The effectiveness of the perturbations is measured using recommendation performance metrics (like NDCG) and fairness metrics (like demographic parity). These metrics help validate the success of the counterfactual interventions in achieving a more equitable distribution of recommendation utility.

### **Systematic Evaluation**

- GNNUERS is systematically evaluated across multiple datasets and models to ensure the robustness and generalizability of its findings. This involves testing the framework under different settings and measuring its impact on both fairness and overall recommendation accuracy.

---

## Counterfactual Concept

The concept of counterfactuality in the GNNUERS framework is implemented through a process of perturbing the original bipartite graph of user-item interactions. Here’s a detailed explanation of how the counterfactual concept is introduced and implemented in their solution:

### **Introduction of Counterfactuality**

1. **Definition of Counterfactual Explanation**:
    - A counterfactual explanation in this context refers to a hypothetical alternative version of the user-item interaction graph. The key idea is to modify this graph in a way that if certain interactions (edges) had not occurred, the outcome (recommendation fairness) would have been different and potentially more equitable.
    - This means identifying specific edges whose removal or modification leads to a decrease in the unfairness metric, which in this case is operationalized through the disparity in utility between different demographic groups.
2. **Conceptual Framework**:
    - Counterfactual reasoning in GNNUERS involves imagining a different state of the world where some user-item interactions are altered. By assessing how these changes affect the fairness of the system, the framework identifies which interactions contribute to unfair outcomes.

### **Implementation of Counterfactuality**

1. **Perturbation Mechanism**:
    - The practical implementation of counterfactuality is through a perturbation mechanism where edges in the bipartite graph are selectively removed or modified. The selection of these edges is guided by a loss function designed to minimize unfairness while keeping the perturbations minimal.
    - Each perturbed version of the graph represents a counterfactual scenario where certain interactions did not happen as they did in the actual data.
2. **Loss Function**:
    - The loss function integrates two components: a fairness term and a sparsity term. The fairness term aims to minimize the utility disparity across demographic groups, and the sparsity term aims to minimize the number of changes made to the graph.
    - By optimizing this loss function, GNNUERS effectively searches for the minimal set of changes (counterfactual perturbations) that would lead to the desired fairness improvements.
3. **Demographic Parity as Fairness Criterion**:
    - Demographic parity is used as a measure of fairness, ensuring that the utility of recommendations is evenly distributed across different groups. The counterfactual reasoning targets this criterion by attempting to equalize the recommendation performance across these groups through structural modifications of the graph.
4. **Iterative Optimization**:
    - The process involves iteratively adjusting the graph structure based on the feedback from the loss function. This iterative approach allows GNNUERS to refine the set of perturbations needed to achieve the fairness goals, simulating different counterfactual scenarios until the optimal configuration is found.
5. **Empirical Validation**:
    - The effectiveness of these counterfactual interventions is then tested empirically on real-world data sets to see if the changes indeed lead to more equitable outcomes. This validation helps confirm whether the identified perturbations correctly pinpoint the interactions contributing to unfairness.

In summary, the counterfactual concept in GNNUERS is implemented through a systematic approach to modifying the graph structure of user-item interactions, aiming to discover minimal changes that could have led to a fairer outcome. This method provides a powerful tool for understanding and mitigating sources of bias in recommender systems.