# 31: Post Processing Recommender Systems with Knowledge Graphs
for Recency, Popularity, and Diversity of Explanations

Features:

Questions:

---

---

## Genearal Summary

This paper, titled "Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations," addresses the enhancement of explanation quality in recommender systems (RS) through the utilization of knowledge graphs (KG). Here’s a summary of the paper:

### **Background and Motivation:**

- **Explainable Recommender Systems**: Traditional explainable RS have focused on explaining recommendations by showing relationships between recommended and previously interacted items. However, the specific properties of explanations, such as recency, popularity, and diversity, have not been sufficiently studied in terms of their impact on explanation quality.

- **Importance of Explanation**: Explanations increase user trust, help users make decisions faster, and improve user satisfaction. The European GDPR also mandates a "right to explanation."

### **Contributions:**

1. **Conceptualization of Explanation Properties**: The paper identifies three key properties influencing explanation quality:
    - **Recency**: How recent the interaction with an item in the explanation is.
    - **Popularity**: The popularity of shared entities (e.g., actors, directors) in the explanation.
    - **Diversity**: The diversity of explanation types across a list of recommendations.
2. **Re-ranking Approaches**: The authors propose re-ranking the recommended items and their explanations to optimize for these properties.
    - **Soft Optimization**: Re-ranks the explanation paths for each recommended item without altering the original list of recommended items.
    - **Weighted Optimization**: Re-ranks both the recommended items and the explanation paths to optimize the explanation properties while considering the recommendation utility.
3. **Evaluation and Results**:
    - **Data Sets**: Experiments were conducted on two public data sets, MovieLens-1M (ML1M) and LastFM-1B (LASTFM).
    - **Metrics**: Explanation quality was assessed using the proposed metrics (recency, popularity, diversity), and recommendation utility was measured using NDCG (Normalized Discounted Cumulative Gain).
    - **Findings**:
        - Significant improvements in explanation quality were observed without substantial losses in recommendation utility.
        - Optimizing for one property often led to gains in others, indicating positive interdependencies between the properties.
        - The proposed re-ranking approaches achieved state-of-the-art results in terms of recommendation utility.
        - The approaches also mitigated unfairness across demographic groups in several cases.

### **Conclusion:**

The paper demonstrates that it is possible to significantly enhance the quality of explanations in RS by optimizing for recency, popularity, and diversity of explanations. The proposed approaches improve user trust and satisfaction while preserving or even enhancing recommendation utility.

### **Future Work:**

- Extending the range of explanation properties.
- Developing novel methods to integrate these properties directly into the model learning process.
- Exploring the generalizability to other domains and datasets.
- Investigating the impact of these approaches on user experience through real-world experiments.

This paper highlights the importance of considering multiple facets of explanation quality and provides robust methods to enhance the explainability of recommender systems, paving the way for more user-centric and transparent recommendation models.

---

## Technical Aspects

### **Key Concepts and Definitions**

1. **Knowledge Graph (KG)**: A KG is a network of entities (like users, products, actors) and the relationships between them (like "watched", "starred in"). It provides a rich structure to represent data.
2. **Explanation Path**: This is a sequence that connects a user to a recommended item through intermediate entities and relationships. For example, "User1 watched Movie1, which was directed by Director1, who also directed Movie2."

### **Explanation Properties**

The authors focus on three key properties to improve the quality of explanations:

1. **Recency**: This property measures how recent the interaction in the explanation path is. For instance, if the explanation path involves a movie the user watched last week, it would be considered recent and more relevant.
2. **Popularity**: This measures how popular the entities in the explanation path are. For example, an explanation that mentions a well-known director is likely to be more meaningful to users.
3. **Diversity**: This looks at how varied the explanation types are in a list of recommendations. If all explanations are similar, it might be less interesting than if they are diverse, showing different types of relationships.

### **Optimization Approaches**

The authors propose two methods to optimize the recommendations based on these properties:

1. **Soft Optimization**: This approach only changes the explanation paths for each recommended product, without changing the order of the recommended products themselves. It selects the best explanation paths according to the desired properties (recency, popularity, or diversity).
2. **Weighted Optimization**: This approach changes both the recommended products and their explanation paths. It balances between keeping the recommendation useful (relevance) and improving the quality of explanations. It re-orders the list of recommended products based on a combination of how relevant the product is and how well the explanation meets the desired properties.

### **Practical Steps**

1. **Identify Explanation Paths**: For each recommended item, identify all possible paths in the KG that can explain why the item is recommended.
2. **Evaluate Explanation Quality**: Assess each path based on recency, popularity, and diversity.
3. **Re-rank Recommendations**:
    - In **soft optimization**, select the best paths for the existing list of recommendations.
    - In **weighted optimization**, adjust both the items in the recommendation list and their explanations to find the best balance between relevance and explanation quality.

### **Outcome**

The optimized recommendations provide better explanations by making them more recent, popular, and diverse, while still keeping the recommendations relevant to the user’s interests. This helps users understand why an item is recommended, increasing their trust and satisfaction with the recommendation system.

---

## Neural Model and Structural Details

### **Neural Model Used**

The paper primarily utilizes the **Policy-Guided Path Reasoning (PGPR)** model, which incorporates reinforcement learning (RL) to navigate the knowledge graph for generating explanations.

**Policy-Guided Path Reasoning (PGPR)**:

- **Reinforcement Learning (RL) Framework**: PGPR uses an RL agent to navigate the knowledge graph and find paths from users to items. The agent learns to select paths that are more likely to lead to relevant and explainable recommendations.
- **Policy Network**: The agent uses a policy network to decide the next step in the path. The policy is trained to maximize the likelihood of selecting paths that lead to useful recommendations.
- **Training**: The agent is trained with a reward signal that combines recommendation relevance and the properties of the paths (recency, popularity, diversity).

---

## Counterfactual Concept