# 28: Path-Specific Counterfactual Fairness

Features:

Questions:

---

The authors introduced Path-Specific Counterfactual Fairness (PSCF) and implemented it using Variational Autoencoders (VAEs) to ensure decisions are fair by altering the influence of sensitive attributes (e.g., gender) along unfair pathways while retaining fair information. They used counterfactual reasoning with abduction-action-prediction steps and variational inference to correct unfair pathways and ensure the independence of latent variables from the sensitive attribute, validated on datasets like UCI Adult and German Credit.

---

## Genearal Summary

The paper addresses the challenge of developing fair decision systems from data that include sensitive attributes affecting decisions through both fair and unfair pathways. The authors propose a counterfactual approach to mitigate the influence of unfair pathways without significantly sacrificing individual-specific information. Here's a summary of the key points:

### **Key Concepts:**

1. **Fairness in Decision Systems**:
    - **Sensitive Attributes**: Attributes like race and gender that can unfairly influence decisions.
    - **Fair vs. Unfair Pathways**: Pathways through which sensitive attributes affect decisions can be either fair (justifiable) or unfair (discriminatory).
2. **Counterfactual Fairness**:
    - This concept ensures fairness by comparing decisions with those in a counterfactual world where the sensitive attribute is different.
    - **Limitations**: Previous definitions of counterfactual fairness consider all effects of sensitive attributes as problematic, which is restrictive.
3. **Path-Specific Counterfactual Fairness (PSCF)**:
    - **Novel Definition**: The decision is fair if it aligns with a counterfactual world where only the unfair pathways are altered.
    - **Application**: Corrects observations adversely affected by sensitive attributes along unfair pathways while retaining fair information.
4. **Implementation with VAE**:
    - **Variational Autoencoder (VAE)**: Utilizes recent advancements in deep learning and approximate inference to handle complex non-linear models.
    - **Latent Variables**: Introduces latent variables to account for unobservable factors, ensuring the fair treatment of individuals by correcting observations during testing.

### **Example Scenario:**

- **Berkeley's Alleged Sex Bias**: Demonstrates how PSCF can handle complex scenarios where gender affects department choice (a fair pathway) but should not affect the admission decision directly.

### **Methodology:**

1. **Graphical Causal Models (GCMs)**:
    - Used to represent and analyze causal relationships.
    - Enables the isolation of unfair causal effects for correction.
2. **Direct and Indirect Effects**:
    - Distinguishes between direct effects (sensitive attribute directly influencing the outcome) and indirect effects (influencing through another variable).
3. **Experiments**:
    - Conducted on datasets like UCI Adult and German Credit to demonstrate the effectiveness of PSCF-VAE in achieving fairness without significant loss in predictive accuracy.

### **Contributions:**

1. **New Definition of Fairness**: More nuanced than existing definitions, allowing for fair pathways to be considered.
2. **Improved Methodology**: The VAE-based approach retains more individual-specific information, enhancing decision accuracy and fairness.
3. **Applicability**: The method is broadly applicable to various complex, non-linear decision systems.

In this paper, "path-specific" refers to focusing on particular causal pathways through which a sensitive attribute (e.g., gender) influences the decision outcome. The method identifies and corrects only the unfair pathways (paths that introduce bias) while preserving the influence through fair pathways. This targeted approach ensures that decisions are fair by altering the unfair causal effects of the sensitive attribute without unnecessarily discarding all its influence.

### **Future Work:**

- The paper suggests future research to relax the requirement of providing the causal model, making the method more broadly applicable without prior knowledge of the data-generation process.

In essence, this paper contributes to the field of fair machine learning by proposing a more refined and applicable approach to mitigating bias in decision systems while preserving accuracy and individual-specific details.

---

## Technical Aspects

---

## Neural Model and Structural Details

---

## Counterfactual Concept

The counterfactual concept is introduced and implemented in their solution through the framework of path-specific counterfactual fairness (PSCF) and the use of Variational Autoencoders (VAEs) to correct unfair pathways while retaining fair information. Here’s a detailed explanation of how the counterfactual concept is introduced and implemented:

### **Introduction of the Counterfactual Concept**

1. **Path-Specific Counterfactual Fairness (PSCF)**:
    - **Definition**: A decision is fair if it matches the decision that would have been made in a counterfactual world where the sensitive attribute (e.g., gender) is altered only along the unfair pathways.
    - **Example**: In the Berkeley gender bias case, PSCF would ensure that a female candidate’s admission decision is fair if it remains the same when assuming the candidate were male along the direct path from gender to admission decision (A → Y).
2. **Graphical Causal Models (GCMs)**:
    - **Causal Paths**: GCMs are used to identify and visualize the causal relationships between variables, distinguishing between fair and unfair pathways.
    - **Directed Acyclic Graphs (DAGs)**: The models ensure that no variable can be its own ancestor, which simplifies the causal analysis.

### **Implementation of the Counterfactual Concept**

1. **Counterfactual Reasoning**:
    - **Abduction**: Estimate the latent variables and noise terms from the observed data.
    - **Action**: Modify the GCM to reflect the counterfactual intervention, where the sensitive attribute is set to a different value along the unfair pathways.
    - **Prediction**: Use the modified GCM to make predictions based on the counterfactual scenario.
2. **Use of Variational Autoencoders (VAEs)**:
    - **Latent Variables**: VAEs introduce latent variables to capture unobserved factors that influence the decision-making process.
    - **Approximate Inference**: Due to the complexity and non-linearity of the models, exact inference is intractable. VAEs provide a framework for approximate inference using variational techniques.
3. **Fair Prediction Mechanism**:
    - **Correction of Descendants**: Correct the variables that are descendants of the sensitive attribute along unfair pathways during testing, while retaining the fair information.
    - **Variational Inference**: Use variational inference to approximate the posterior distributions of the latent variables, ensuring that they do not depend on the sensitive attribute.