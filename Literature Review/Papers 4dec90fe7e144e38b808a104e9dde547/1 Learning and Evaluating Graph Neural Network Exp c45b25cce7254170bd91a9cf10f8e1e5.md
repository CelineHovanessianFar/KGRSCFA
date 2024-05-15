# 1: Learning and Evaluating Graph Neural Network Explanations based on Counterfactual and Factual Reasoning

Features:

- Model Agnostic
- Addresses the evaluation of explanations
- Use with GNNS

### **Explanation Generation: CF2 Framework**

### **Counterfactual and Factual Reasoning**

- **Factual Reasoning**: This seeks to identify a sub-graph whose information is sufficient to maintain the same prediction as the whole original graph. It focuses on ensuring that the presence of certain features or edges leads to the predicted outcome.
- **Counterfactual Reasoning**: In contrast, this identifies necessary elements of the graph—features or edges whose absence would change the outcome. This reasoning focuses on determining the minimal set of critical components.

### **Optimization Problem Formulation**

The CF2 framework formulates the explanation task as an optimization problem where both counterfactual and factual reasoning are incorporated into the objective function. This dual approach aims to find a sub-graph that is both minimally sufficient and necessarily complete, ensuring that the explanation includes all and only those components that significantly influence the prediction.

- **Objective**: Minimize the complexity of the explanation (measured by the number of edges and features used) while ensuring the explanation is sufficiently strong (effective at predicting the correct outcome).
- **Constraints**: The optimization includes constraints to ensure that the explanation meets the criteria for both factual and counterfactual conditions.

### **Evaluation of Explanations**

Since ground-truth explanations are not available for most real-world graphs, evaluating the quality of generated explanations poses a significant challenge. The paper addresses this by proposing two metrics based on the notions of necessity and sufficiency:

- **Probability of Necessity (PN)**: Measures how often the removal of the explained sub-graph results in a different prediction, thus assessing the necessity of the included components.
- **Probability of Sufficiency (PS)**: Assesses whether the sub-graph alone can lead to the same prediction as the entire graph, thus evaluating its sufficiency.

### **Implementation Details**

- **Loss Function**: The paper uses a novel loss function that balances the trade-off between the explanation's complexity and its strength. This function includes terms for the norm (to control complexity) and a contrastive loss (to ensure the explanation meets factual and counterfactual conditions).
- **Relaxation Techniques**: To make the optimization tractable, the CF2 framework employs relaxation techniques where hard binary decisions (whether to include an edge/feature) are approximated by continuous values, which are then thresholded to make final decisions.

### **Practical Application**

The approach is validated on several datasets where the CF2 framework consistently outperforms existing methods by generating explanations that are more aligned with both necessary and sufficient conditions. This not only enhances the transparency of GNNs but also supports the development of more reliable and interpretable AI systems.

By combining these technical elements, the CF2 framework offers a comprehensive and robust method for both generating and evaluating explanations of GNN predictions, advancing the field of explainable AI.