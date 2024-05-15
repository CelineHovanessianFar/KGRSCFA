# 24: CF-GNNExplainer: Counterfactual Explanations for Graph Neural
Networks

Features:

Questions:

---

The authors developed CF-GNNExplainer, a method for generating counterfactual explanations for Graph Neural Networks (GNNs) by iteratively removing edges from the graph's adjacency matrix. This approach identifies the minimal perturbations necessary to change the GNN's prediction for a given node, ensuring explanations are both concise and accurate. The method was evaluated on three datasets, demonstrating high accuracy with minimal edge deletions.

---

## Genearal Summary

The paper "CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks" focuses on providing counterfactual (CF) explanations for Graph Neural Networks (GNNs). Unlike existing methods that generate subgraphs relevant to a particular prediction, CF-GNNExplainer aims to understand how a prediction can be changed to achieve an alternative outcome. The proposed method generates CF explanations by making minimal perturbations, specifically edge deletions, to the input graph data to change the prediction.

**Key Points:**

1. **Introduction:**
    - The paper addresses the growing demand for explainable AI, particularly for GNNs used in real-world applications.
    - Traditional methods focus on relevant subgraphs for a given prediction but are not counterfactual in nature.
    - CF explanations answer "What changes would lead to a different outcome?"
2. **Methodology:**
    
    
    - **CF-GNNExplainer:** The method iteratively removes edges from the graph's adjacency matrix using matrix sparsification techniques until the prediction changes.
    - The goal is to find the minimal number of edge deletions that result in a different prediction, ensuring that the explanation is concise and impactful.
3. **Evaluation:**
    - The method is evaluated on three public datasets (tree-cycles, tree-grids, and ba-shapes) using metrics such as fidelity, explanation size, sparsity, and accuracy.
    - CF-GNNExplainer achieves high accuracy (at least 94%) while removing fewer than three edges on average, indicating that the removed edges are crucial for the original predictions.
4. **Comparison:**
    - The paper compares CF-GNNExplainer with several baselines, including random perturbation, 1-hop neighborhood methods, and GNNExplainer (an existing GNN explanation method).
    - CF-GNNExplainer outperforms these baselines in terms of producing minimal, accurate CF explanations.
5. **Results:**
    - CF-GNNExplainer generates CF examples for the majority of instances across the datasets.
    - It achieves minimal explanations with high fidelity and accuracy, showing that it effectively identifies crucial edges for predictions.
6. **Societal Impact and Future Work:**
    - The paper highlights the importance of considering the context in which CF explanations are used and the need for rigorous evaluation protocols for XAI methods.
    - Future work includes extending the method to accommodate graph classification tasks, incorporating node feature perturbations, and conducting user studies to assess the practical utility of CF-GNNExplainer.

Overall, the paper presents a novel and effective method for generating CF explanations for GNNs, contributing to the broader field of explainable AI and addressing the specific need for understanding and modifying GNN predictions.

---

## Technical Aspects

### **Problem Formulation**

The goal is to generate counterfactual (CF) explanations for a given node's prediction by a Graph Neural Network (GNN). Specifically, the method seeks to identify the minimal changes to the graph's structure (i.e., edge deletions) that result in a different prediction for the node.

### **Graph Neural Networks (GNNs)**

GNNs operate on graph structures, where nodes represent entities and edges represent relationships between them. A typical GNN updates node representations by aggregating features from neighboring nodes through multiple layers. The final node representations are then used for tasks such as node classification.

### **CF-GNNExplainer Method**

### **1. Adjacency Matrix Perturbation**

- **Objective**: Find a perturbation matrix *P* such that the GNN's prediction for the perturbed graph differs from the original prediction.
- **Process**:
    - The original adjacency matrix *Av* for the node's subgraph is perturbed to *Av*ˉ=*P*⊙*Av*, where *P* is a binary matrix indicating which edges to keep (1) or remove (0), and ⊙ denotes element-wise multiplication.

### **2. Counterfactual Generating Model**

- **Function 𝑔*g***: This function generates counterfactual examples by applying the perturbation *P* to the adjacency matrix *Av*. It shares the same weight parameters *W* as the original GNN model *f*.
- **Modified Adjacency Matrix**: The perturbed adjacency matrix *Av*ˉ is calculated without affecting the self-loops (which are preserved by adding an identity matrix *I*).

### **3. Loss Function Optimization**

The method optimizes a loss function that balances the prediction change and the minimality of the perturbation:

![Untitled](24%20CF-GNNExplainer%20Counterfactual%20Explanations%20for%20b27634bc42d24b54a4be52b12b2b99d6/Untitled.png)

- **Prediction Loss 𝐿pred*L*pred**: Encourages the perturbed prediction to differ from the original prediction.
- **Distance Loss 𝐿dist*L*dist**: Ensures the perturbation is minimal by penalizing large changes in the adjacency matrix.

### **4. Optimization Process**

- **Initialization**: Start with a perturbation matrix *P*^ initialized to all ones (indicating no perturbations initially).
    
    𝑃^
    
- **Iterations**: Over multiple iterations, update *P*^ to minimize the loss function. Apply a sigmoid transformation followed by a threshold to convert *P*^ into a binary perturbation matrix *P*.
- **Gradient Descent**: Use gradient-based optimization (e.g., SGD) to update *P*^ based on the loss gradient.

### **5. Algorithm**

The core steps of the CF-GNNExplainer algorithm are summarized as follows:

1. **Obtain Original Prediction**: Get the GNN prediction *f*(*v*) for the node *v*.
2. **Initialize Perturbation Matrix**: Set *P*^ to all ones.
3. **Iterative Optimization**:
    - For each iteration:
        - Compute the binary perturbation matrix *P* by thresholding *P*^.
        - Generate the counterfactual example *v*ˉ using *P*.
        - Update *P*^ by minimizing the loss function.
4. **Return Optimal CF Explanation**: After completing the iterations, return the perturbation that results in the minimal valid counterfactual example.

### **Evaluation**

The method is evaluated using several metrics:

- **Fidelity**: Proportion of nodes where the prediction changes (lower is better for CF explanations).
- **Explanation Size**: Number of edges removed (smaller is better).
- **Sparsity**: Proportion of edges in the subgraph that are removed (higher is better).
- **Accuracy**: Proportion of correct CF explanations, where only crucial edges are removed.

### **Key Advantages**

- **Minimal Perturbations**: CF-GNNExplainer finds the smallest changes needed to alter the prediction, leading to concise explanations.
- **High Accuracy**: The method accurately identifies edges crucial to the original prediction, ensuring the explanations are meaningful and reliable.

In summary, CF-GNNExplainer provides a novel approach to generating CF explanations for GNNs by leveraging edge deletions to identify minimal and accurate changes that alter node predictions, enhancing the interpretability of GNN models.

---

## Neural Model and Structural Details

---

## Counterfactual Concept