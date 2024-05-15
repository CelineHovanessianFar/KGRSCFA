# 10: Causal Inference for Knowledge Graph based Recommendation

Features:

Questions:

1. What does this mean: This score forms a basis for the recommendation system.
2. does ‘To implement counterfactual reasoning, the model calculates what the similarity scores (S) would look like if the direct effects of the interacted attributes (A) were removed. “ no problem with the embeddings?

---

The KGCR model integrates causal inference into a knowledge graph-based recommendation system by constructing a causal graph to systematically identify and mitigate biases in user preferences and similarity scoring. It employs Graph Convolutional Networks (GCN) to refine embeddings and introduces counterfactual reasoning to adjust similarity scores by removing the direct effects of user-interacted attributes, leading to more accurate and unbiased recommendations.

---

## Genearal Summary

This paper introduces the Knowledge Graph-based Causal Recommendation (KGCR) model, which integrates causal inference into knowledge graph-based recommender systems to address two main challenges:

1. **Learning User Preferences on Attributes**: Previous models often failed to accurately represent user preferences regarding specific attributes within a knowledge graph, such as item features (like brand or genre). KGCR addresses this by using causality tools to identify and mitigate the effects of structural biases in the knowledge graph that may obscure true user preferences.
2. **Bias in Similarity Scores**: The interaction of a user with certain attributes can bias similarity scores, leading to inaccurate recommendations. KGCR employs counterfactual reasoning to adjust these scores, aiming to reduce bias and improve the accuracy of the recommendations.

The model operates by constructing a causal graph that maps the relationships and influences between users, items, and attributes. It then performs interventions in this graph to disentangle the influence of structure from user preferences and to correct for bias in similarity scoring.

The paper validates KGCR's effectiveness through extensive experiments across multiple datasets (Amazon-book, LastFM, and Yelp2018), showing that it outperforms several state-of-the-art models. The approach not only enhances recommendation accuracy but also provides a framework for understanding how different factors within a knowledge graph influence user-item interactions, making recommendations more interpretable.

---

## Technical Aspects

The technical solution provided by the KGCR model involves several key components that leverage causal inference to enhance knowledge graph-based recommendation systems. Here's a breakdown of these components:

### **1. Causal Graph Construction**

KGCR begins by constructing a causal graph that represents the causal relationships among different variables in the recommendation system. These variables include:

- **U (User Preferences on Attributes)**: Represents how user preferences align with certain attributes in items.
- **I (Item Representations)**: Describes items based on their attributes extracted from the knowledge graph.
- **A (User’s Interacted Attributes)**: Represents the attributes of items with which a user has previously interacted.
- **K (Structural Information)**: Includes the structural information of the knowledge graph that could influence attribute exposure to users.
- **S (Similarity Scores)**: Measures the similarity between user preferences and item attributes, used for making recommendations.

### **2. Causal Intervention**

To mitigate the influence of confounders such as the structural information (K) that may affect the accurate learning of user preferences (U) for attributes (A), KGCR performs causal interventions. This is done using do-calculus, where certain edges in the causal graph are "cut" to simulate a scenario where the confounder’s influence is removed. This helps in obtaining a more deconfounded estimation of user preferences.

### **3. Counterfactual Inference for Bias Reduction**

KGCR uses counterfactual reasoning to address biases in the similarity scores between users and items. This is achieved by estimating what the similarity score would have been under different hypothetical scenarios (counterfactuals) where the user had not interacted with certain attributes. This helps in isolating and removing the biased effect of these attributes on the similarity score.

### **4. Model Implementation and Optimization**

- **Knowledge Graph Embeddings**: Initial item and attribute representations are learned using embeddings from the knowledge graph.
- **Graph Convolutional Networks (GCN)**: KGCR employs GCNs to refine these embeddings by propagating and aggregating information across the graph structure, enhancing the attribute and item representations with contextual graph-based information.
- **Debiasing Similarity Scores**: The model computes debiased user-item similarity scores by combining counterfactually adjusted attribute effects with base collaborative filtering signals.

### **5. Loss Functions and Training**

The training of KGCR involves optimizing a loss function that considers both the traditional collaborative filtering loss and a novel counterfactual loss component that aims to minimize the discrepancy caused by biased interactions. The loss function integrates:

- **Bayesian Personalized Ranking (BPR) Loss**: Optimizes for the correct ranking of items for each user, promoting higher ranks for interacted items over non-interacted ones.
- **Attribute-Based Similarity Loss**: Ensures that the similarity scores reflect true attribute preferences by penalizing scores that do not align with user interactions.

By integrating these techniques, KGCR aims to provide more accurate and fair recommendations by understanding and adjusting for the underlying causal relationships within the data. This approach not only improves recommendation quality but also contributes to the transparency and interpretability of the model’s decisions.

---

## Neural Model and Structural Details

In the KGCR framework, several neural modeling techniques and data representation strategies are employed to optimize the recommendation system. Here are the technical details regarding the neural model usage and data representation in the KGCR:

### **Neural Model Usage**

1. **Graph Convolutional Networks (GCN)**:
    - **Role**: GCNs are used to process the knowledge graph data, which helps in capturing the structured relationships between entities (like items and attributes) effectively.
    - **Implementation**: The user and item embeddings are refined through multiple layers of graph convolutions, where each layer aggregates neighborhood information to update the node embeddings. This process enhances the representations by incorporating contextual information from the knowledge graph.
2. **Embedding Techniques**:
    - **TransE for Initialization**: Initial embeddings for entities and relations in the knowledge graph are learned using the TransE model. TransE is a translation-based embedding technique that models relationships by interpreting them as translations operating on the embeddings of the entities.
    - **Attribute and Item Representation**: Post initial embedding with TransE, further refinement is done using GCN layers, which help to blend the embeddings with the structural and relational context from the knowledge graph.
3. **Inner Product for Similarity Calculation**:
    - **Use**: After obtaining refined embeddings from the GCNs, the similarity between user embeddings and item embeddings is calculated using the inner product. This score forms a basis for the recommendation system.

### **Data Representation**

1. **User and Item Embeddings**:
    - Users and items are represented through embeddings derived from the knowledge graph. The embeddings capture not just the inherent properties of users and items but also their contextual relationships derived from the graph structure.
2. **Knowledge Graph Structure**:
    - **Entities and Relationships**: The knowledge graph comprises entities (items, attributes) and relationships between them. Each entity and relationship is initially embedded using TransE, and these embeddings are further refined through GCNs.
    - **Graph Layers**: The GCN layers operate over the structured data of the knowledge graph, processing entities and their relationships in layers to refine their representations based on the connectivity patterns.

### **Technical Details of Model Training and Optimization**

1. **Loss Functions**:
    - **Bayesian Personalized Ranking (BPR) Loss**: This loss is used to optimize the ordering of items such that items interacted with by a user are ranked higher than those not interacted with. It helps in learning personalized ranking for users.
    - **Attribute Similarity Adjustment**: Additional loss components are introduced to adjust for the biases in attribute-based similarity calculations, ensuring that the recommendations reflect true user preferences and are not merely driven by biased or spurious attribute interactions.
2. **Counterfactual Reasoning**:
    - Counterfactual reasoning is applied to calculate what the similarity scores between users and items would be if certain biases (from user’s previously interacted attributes) were not present. This helps in assessing the direct and indirect effects of attributes on the recommendations, allowing for a debiased estimation of user-item affinities.
3. **Optimization**:
    - **Parameter Optimization**: The model parameters are optimized using stochastic gradient descent with the Adam optimizer, which is effective for handling sparse data and large-scale parameter spaces typical of knowledge graphs and recommendation systems.

This combination of neural network models, sophisticated embedding techniques, and advanced training strategies enables KGCR to effectively handle the complexities of knowledge graph-based recommendation systems while addressing bias and improving interpretability.

---

## Counterfactual Concept

In the KGCR framework, counterfactual reasoning is a critical method used to handle biases in the recommendation process, particularly those stemming from a user's previously interacted attributes. Here’s how the counterfactual concept is introduced and implemented in their solution:

### **Introduction of Counterfactual Reasoning**

Counterfactual reasoning in the context of KGCR is employed to understand and adjust for the potential bias that might arise when a user’s historical interactions with certain attributes influence the similarity scores used in recommendations. The goal is to determine what the similarity score between a user and an item would have been if the user had not interacted with certain biased attributes.

### **Implementation Steps**

1. **Constructing the Causal Graph**:
    - **Causal Variables**: The causal graph includes variables such as user preferences (U), item attributes (I), interacted attributes (A), and similarity scores (S).
    - **Causal Relationships**: Relationships among these variables are mapped, particularly focusing on how interacted attributes (A) might influence both user preferences (U) and similarity scores (S).
2. **Identifying Direct and Indirect Effects**:
    - **Direct Effects (DE)**: These are the effects that the interacted attributes (A) have directly on the similarity scores (S).
    - **Indirect Effects (IE)**: These effects are mediated through other variables, such as the influence of interacted attributes on user preferences (U), which in turn affect similarity scores (S).
3. **Counterfactual Intervention**:
    - **Modification of Attribute Effects**: To implement counterfactual reasoning, the model calculates what the similarity scores (S) would look like if the direct effects of the interacted attributes (A) were removed. This involves estimating the similarity score under a hypothetical or counterfactual scenario where the user interacts with the item but without the bias introduced by specific attributes.
4. **Calculation of Counterfactual Scores**:
    - **Counterfactual Inference Formula**: The main counterfactual inference involves computing the "Natural Direct Effect" (NDE) and the "Total Effect" (TE) of the interacted attributes on the similarity scores.
    - **Formula Implementation**: The model computes the difference between the factual and counterfactual similarity scores. Mathematically, this can be expressed as:
        
        𝑇𝐸−𝑁𝐷𝐸=(𝑆factual−𝑆counterfactual)*TE*−*NDE*=(*S*factual−*S*counterfactual)
        
    
    where 𝑇𝐸*TE* is the effect of the interacted attributes under the actual user interactions, and 𝑁𝐷𝐸*NDE* is the effect of the same attributes if the direct path from the attributes to the similarity scores were blocked.
    
5. **Adjustment of Scores**:
    - **Final Score Calculation**: The adjusted, debiased similarity scores are then calculated by subtracting the estimated direct effects from the total observed effects, yielding a score that reflects what the user's preferences would likely be in the absence of the identified biases.
6. **Optimization**:
    - **Loss Function Incorporation**: The debiased similarity scores are integrated into the model's loss function to optimize the overall recommendation accuracy, ensuring that the training process minimizes these biases effectively.

Through these steps, KGCR leverages counterfactual reasoning to systematically address and mitigate the influence of bias in recommendations, leading to more accurate and fair user-item matching in the recommendation systems.