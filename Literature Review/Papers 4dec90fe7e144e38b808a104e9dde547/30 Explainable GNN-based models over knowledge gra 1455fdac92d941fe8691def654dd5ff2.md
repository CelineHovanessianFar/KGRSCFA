# 30: Explainable GNN-based models over knowledge graphs

Features:

Questions:

---

The authors developed Monotonic Graph Neural Networks (MGNNs) that transform knowledge graphs (KGs) in a way that can be explained using Datalog rules. They encode KGs into numeric feature vectors, process them with MGNNs, and decode the results back to KGs, ensuring that each transformation corresponds to a logical inference step, thus providing symbolic explanations for the predictions.

---

## Genearal Summary

The paper proposes a new approach to using Graph Neural Networks (GNNs) for transforming knowledge graphs (KGs) while ensuring that the predictions can be explained symbolically. The key contributions are as follows:

1. **Introduction of Monotonic GNNs (MGNNs)**: The paper presents a novel class of GNNs that are designed to ensure that the transformations they perform on graph data can be explained using logical rules in the Datalog formalism.
2. **Encoding and Decoding Process**: The approach involves encoding an input KG into a graph with numeric feature vectors, processing this graph using an MGNN, and then decoding the result back into an output KG.
3. **Equivalence to Datalog Rules**: The MGNNs are constructed to ensure that their transformation process is equivalent to a round of application of a set of Datalog rules. This means that for any MGNN, it is possible to extract a set of Datalog rules that fully characterize the transformation.
4. **Application and Evaluation**: The approach is applied to classification tasks in knowledge graph completion. The paper demonstrates that the system can provide explanations for its predictions and shows competitive performance compared to state-of-the-art systems.
5. **Detailed Proofs and Theoretical Foundations**: The paper includes formal proofs to show the equivalence between MGNN transformations and Datalog rule applications, ensuring that the proposed method is theoretically sound.

Overall, this work bridges the gap between GNN-based approaches and symbolic reasoning, providing a framework that leverages the strengths of both machine learning and logical inference for tasks involving knowledge graphs.

---

## Technical Aspects

### **1. Graph Neural Networks (GNNs) and Knowledge Graphs (KGs)**

**Graph Neural Networks (GNNs)**: These are a type of neural network designed to work directly with the graph structure of data. In a GNN, each node in the graph (representing an entity) is associated with a feature vector. The network updates these feature vectors by aggregating information from the node's neighbors, layer by layer.

**Knowledge Graphs (KGs)**: These are structured representations of knowledge, where entities (nodes) are connected by relationships (edges). Examples include user-item interactions in a recommender system or facts in a database.

### **2. Monotonic GNNs (MGNNs)**

The authors propose a special class of GNNs called Monotonic GNNs (MGNNs). The key feature of MGNNs is that they ensure monotonicity: if the values of input features increase, the output features do not decrease. This property is crucial for ensuring that the GNN’s predictions can be explained logically.

### **3. Encoding and Decoding Process**

**Encoding**: The process starts by converting the input KG into a graph where each node and edge is labeled with numeric feature vectors. Each type of relationship and entity attribute is mapped to specific positions in these feature vectors.

**Processing**: This encoded graph is then processed by the MGNN. The MGNN updates the feature vectors of each node based on the information from its neighbors. The updates are performed in a way that preserves the monotonicity property.

**Decoding**: After processing, the updated feature vectors are converted back into the original KG format. This involves interpreting the numeric values to reconstitute the entities and relationships.

### **4. Symbolic Explanations with Datalog**

**Datalog**: This is a logical formalism used for defining rules and querying databases. It allows expressing logical rules in the form of "if-then" statements, which can be applied to derive new facts from existing ones.

**Equivalence to Datalog Rules**: The MGNN is designed so that its transformation of the KG can be directly mapped to the application of a set of Datalog rules. This means that each prediction made by the MGNN can be explained as the result of applying specific logical rules to the input data.

### **5. Training and Evaluation**

**Training**: The MGNN is trained using examples of input-output pairs (i.e., incomplete and completed KGs). The training process involves learning the parameters of the MGNN to accurately transform the input KG into the output KG.

**Evaluation**: The approach is evaluated on tasks like KG completion, where the goal is to predict missing relationships in the KG. The performance of the MGNN-based method is compared to other state-of-the-art techniques. Importantly, the authors also demonstrate that the rules extracted from the MGNN can explain the predictions, enhancing interpretability.

### **6. Benefits and Applications**

**Interpretability**: One of the major advantages of this approach is that it provides clear, logical explanations for its predictions, which is a significant improvement over traditional GNNs that operate as black boxes.

**Applications**: The technique is applicable to various domains that use KGs, such as recommendation systems, fraud detection, and data integration. By providing symbolic explanations, it helps in building trust and ensuring compliance with norms and fairness standards.

In summary, the authors' solution integrates the strengths of GNNs for learning from graph-structured data with the interpretability of symbolic logic, offering a powerful tool for tasks involving KGs.

---

## Neural Model and Structural Details

---

## Counterfactual Concept