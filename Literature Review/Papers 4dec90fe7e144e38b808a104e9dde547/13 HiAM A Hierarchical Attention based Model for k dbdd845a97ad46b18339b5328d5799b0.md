# 13: HiAM: A Hierarchical Attention based Model for knowledge graph multi-hop reasoning

Features:

Questions:

---

The authors developed HiAM, a model that utilizes bidirectional LSTM and hierarchical attention mechanisms to process and reason over knowledge graphs. They extract and combine both predecessor and connection paths between entity pairs, embedding these paths to represent structural data. The model applies entity/relation-level and path-level attention to weigh the importance of elements within paths, using this refined information to predict and explain multi-hop reasoning tasks more effectively.

---

## Genearal Summary

This paper introduces a novel Hierarchical Attention based Model (HiAM) for multi-hop reasoning in knowledge graphs, which are used in tasks like question answering and recommender systems. A significant challenge in multi-hop reasoning is effectively synthesizing the structural information (paths) in knowledge graphs for deeper reasoning. Traditional methods often focus on connection paths between entity pairs but tend to overlook predecessor paths and fail to weight the importance of entities and relations within these paths appropriately.

HiAM addresses these limitations by incorporating predecessor paths, which provide semantic contexts that precede connection paths, thus offering a more accurate representation of entities involved. The model also employs a hierarchical attention mechanism that considers the contributions of different granularities within the paths, such as entity/relation-level and path-level features, to enhance the reasoning process.

The paper demonstrates that by combining predecessor paths with connection paths and applying hierarchical attention, HiAM can effectively enhance path representations and discern the varying impacts of different entities and relations on reasoning. Experimental results across three benchmark datasets show that HiAM achieves competitive performance compared to existing methods. Furthermore, the model can explain its predictions by selecting the most significant path, providing greater interpretability.

Overall, this approach not only improves the accuracy of reasoning over knowledge graphs but also enhances the explainability of the results, which is crucial for applications in real-world scenarios where understanding the basis of decisions made by AI systems is essential.

---

## Technical Aspects

The technical solution proposed in the paper, the Hierarchical Attention based Model (HiAM), is designed to address multi-hop reasoning in knowledge graphs by enhancing the use of paths and paying attention to the contributions of different entities and relations. Here's a detailed breakdown of the key components and technical aspects of HiAM:

### **1. Path Extraction**

HiAM starts by extracting two types of paths from the knowledge graph for each entity pair:

- **Predecessor Paths:** These paths provide context and semantic information about the head entity before it connects to another entity. They are used to enrich the representation of the head entity, offering more depth to the initial reasoning state.
- **Connection Paths:** These are direct paths between the entity pair in question and are typically the focus of traditional reasoning methods.

The model combines these paths, using both to feed into the attention mechanisms. This dual-path approach allows the model to have a richer context and more accurate semantics for entities at the beginning of the reasoning process.

### **2. Hierarchical Attention Mechanism**

After the paths are extracted, HiAM applies a hierarchical attention mechanism in two layers:

- **Entity/Relation-Level Attention:** This layer of attention assesses the importance of each entity and relation within a path. Not all entities and relations contribute equally to the reasoning outcome; thus, this attention mechanism helps to weigh their importance based on their contribution to the reasoning task. It helps to filter out noise and focus on the most informative parts of each path.
- **Path-Level Attention:** This higher-level attention aggregates the information from multiple paths, evaluating the significance of each path in the context of the overall reasoning task. It helps to determine which paths are most likely to lead to a correct inference by focusing on paths that are more informative or relevant.

### **3. Fusion of Multi-Granularity Features**

The features processed through the entity/relation-level attention are then fed into the path-level attention. This hierarchical structure allows the model to analyze impacts at different granularities, integrating detailed insights from entities and relations with broader insights from entire paths.

### **4. Prediction and Explainability**

The final step involves fusing the multi-granularity features to predict the correct answers (relations or entities) for the given queries. HiAM goes a step further by selecting the most significant path as an explanation for the predicted answers. This feature is crucial for applications requiring transparency in AI decisions, as it provides a clear and understandable rationale for each prediction.

### **5. Training and Optimization**

The model is trained to maximize the distinction between correct (positive samples) and incorrect (negative samples) predictions by minimizing a margin-based loss function. This training process also involves standard techniques like backpropagation and optimization through Adam optimizer.

### **Summary**

HiAM leverages advanced techniques in neural network architectures and attention mechanisms to handle the complexity of reasoning over large-scale knowledge graphs. By incorporating both predecessor and connection paths and applying a hierarchical attention mechanism, HiAM not only improves the reasoning capabilities over knowledge graphs but also provides a method to make these processes interpretable and justifiable.

---

## Neural Model and Structural Details

The HiAM model employs several neural network techniques and data representation methods to enhance its multi-hop reasoning capabilities over knowledge graphs. Here’s a deeper dive into the neural models used and how data is represented within HiAM:

### **1. Neural Model Used**

### **Bidirectional Long Short-Term Memory (Bi-LSTM)**

- **Role in HiAM:** The Bi-LSTM is crucial in encoding the sequences of entities and relations within each path. By processing information in both forward and backward directions, the Bi-LSTM captures the contextual relationships between entities and relations over the paths, providing a richer and more dynamic representation of the sequential data in the paths.
- **Benefits:** This bidirectional approach allows the model to capture both past (backward) and future (forward) dependencies in data, which is particularly useful in understanding the sequence of entities and relations that lead to a particular inference.

### **2. Data Representation**

### **Embedding Layer**

- **Purpose:** Each entity and relation in the knowledge graph is initially represented as a low-dimensional vector. These embeddings are crucial as they transform sparse, high-dimensional categorical data into a dense, continuous vector space where similar entities and relations are positioned closer together.
- **Initialization:** The embeddings are randomly initialized and then refined during training through backpropagation.

### **Path Representation**

- **Combination of Embeddings:** Each path, comprising a sequence of entities and relations, is represented by combining the embeddings of its constituent entities and relations. This combination could be a simple concatenation or more complex operations that blend the features of entities and relations in the path.
- **Processing:** The combined embeddings of paths are then processed through the Bi-LSTM to capture the sequential interdependencies of entities and relations along the paths.

### **3. Attention Mechanisms**

### **Hierarchical Attention Network**

- **Entity/Relation-Level Attention:**
    - **Mechanism:** This attention layer computes attention scores for each entity and relation within a path, determining their importance to the reasoning task at hand.
    - **Output:** The output is a weighted sum of the LSTM-encoded features of entities and relations, where weights are determined by their respective attention scores.
- **Path-Level Attention:**
    - **Mechanism:** After entity/relation-level attention, the path-level attention aggregates the representations of different paths, evaluating and emphasizing those more relevant to answering the query.
    - **Output:** The model computes a final representation for each entity pair by aggregating over the different path representations with attention scores that reflect the significance of each path in the reasoning process.

### **4. Interaction and Prediction**

### **Interaction Layer**

- **Function:** Post attention, the interaction layer models the relationship between the aggregated path representation and the query relation (in case of predicting relations) using vector operations.
- **Techniques Used:** Element-wise product operations and multi-layer perceptrons (MLP) are utilized to compute the final interaction scores, which serve as the basis for predictions.

### **Prediction Function**

- **Definition:** The prediction function estimates the probability of a candidate relation being correct given the entity pair and the aggregated path information.
- **Activation:** A sigmoid function is applied to the output of the interaction layer to derive probabilities, facilitating binary classification (e.g., relation exists or does not exist).

### **5. Training and Optimization**

### **Objective Function**

- **Margin-Based Loss:** The model uses a margin-based loss function during training, which emphasizes increasing the distance (margin) between the scores of positive and negative examples to improve classification accuracy.
- **Regularization:** Techniques like L2 regularization are used to prevent overfitting and ensure the generalization ability of the model.

### **Optimization**

- **Optimizer:** The Adam optimizer, known for its efficiency in handling sparse gradients and adaptive learning rate capabilities, is employed to minimize the loss function.

### **Conclusion**

HiAM incorporates advanced neural network architectures, sophisticated embedding techniques, and innovative attention mechanisms to effectively process and reason over complex structured data in knowledge graphs. These components work in concert to enhance the model's reasoning performance and its ability to provide interpretable results.

---

## Counterfactual Concept