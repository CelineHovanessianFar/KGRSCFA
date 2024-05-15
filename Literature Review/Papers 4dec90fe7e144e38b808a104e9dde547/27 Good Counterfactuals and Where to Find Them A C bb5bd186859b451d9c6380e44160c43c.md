# 27: Good Counterfactuals and Where to Find Them:
A Case-Based Technique for Generating Counterfactuals
for Explainable AI (XAI)

Features:

Questions:

---

The authors developed a case-based reasoning method to generate counterfactual explanations by identifying pairs of minimally differing cases (explanation cases) in a dataset. They then reused these pairs to construct new counterfactuals for unseen queries, adjusting feature values iteratively to ensure class changes, thereby improving the explanatory coverage and plausibility of generated counterfactuals.

---

## Genearal Summary

This paper discusses a novel approach to generating counterfactual explanations for Explainable AI (XAI) using Case-Based Reasoning (CBR). Counterfactual explanations are favored for their causal informativeness and compliance with legal regulations like GDPR. They are created by altering problem features until a class-change is achieved, such as suggesting to a loan applicant that a slightly lower loan amount would have been approved.

However, current techniques face challenges in generating "good" counterfactuals, which are sparse (modifying few features) and plausible. The paper shows that many commonly used datasets have few naturally occurring good counterfactuals. To address this, the authors propose a new case-based method that leverages existing patterns of good counterfactuals within a case-base to generate new, analogous counterfactuals for explanation purposes.

The method involves:

1. Identifying pairs of cases (explanation cases or XCs) where one case (the counterfactual) differs minimally from the other (the query).
2. Using these XCs to guide the generation of new counterfactuals for unseen queries.
3. Adapting the generated counterfactuals if necessary to ensure they belong to a different class than the query.

Experiments demonstrate that this technique significantly improves the ability to find good counterfactuals in various datasets, enhancing the explanatory coverage and competence of case-bases.

Key contributions of the paper include:

- A novel case-based approach to counterfactual generation.
- Introduction of the concept of explanatory competence.
- Empirical evidence showing improved counterfactual potential and coverage using the proposed technique.

Overall, the paper argues for the effectiveness of a CBR approach in generating meaningful and actionable counterfactual explanations, which are crucial for the transparency and trustworthiness of AI systems.

---

## Technical Aspects

The technical solution proposed in the paper involves a method to generate counterfactual explanations using Case-Based Reasoning (CBR). Here’s a detailed breakdown of their approach:

### **Key Concepts**

1. **Counterfactual Explanations**: These are explanations that show how a different outcome could have been achieved by changing some features of the input. For example, a loan application rejected due to insufficient salary might be explained by saying, "If your salary were $X higher, your loan would have been approved."
2. **Good Counterfactuals**: These are counterfactuals that are sparse (modify few features) and plausible (make realistic changes).
3. **Case-Based Reasoning (CBR)**: This is a method that solves new problems based on the solutions of similar past problems. For XAI, CBR involves using past cases to explain new ones.

### **Technical Steps**

### **1. Explanation Cases (XC)**

- **Explanation Cases (XC)**: An explanation case is a pair of cases: one is the original case (query) and the other is a good counterfactual (minimal difference, different outcome).
- **XC Case-Base**: The collection of all explanation cases in the dataset.

### **2. Building the XC Case-Base**

- Identify pairs of cases in the dataset where one can serve as a counterfactual for the other.
- Ensure the counterfactual case differs by no more than 2 features from the original case.

### **3. Generating Counterfactuals for New Queries**

When a new query (unseen case) needs an explanation:

1. **Retrieval**: Identify the explanation case from the XC case-base that is most similar to the query.
2. **Reuse**: Construct a new counterfactual for the query by combining features from the query and the identified explanation case.
    - **Match-Features**: Features that are the same between the query and the explanation case are copied to the new counterfactual.
    - **Difference-Features**: Features that differ between the query and the explanation case are also applied to the new counterfactual.
3. **Validation**: Check if the newly constructed counterfactual has a different class than the query using the underlying machine learning model.
4. **Adaptation (if needed)**: If the newly constructed counterfactual does not result in a different class:
    - Iterate over the nearest neighbors of the query.
    - Modify the difference features until a valid counterfactual is found (one that results in a different class).

### **4. Assessing Explanatory Competence**

- **Explanatory Competence**: The fraction of query cases that can be explained using good counterfactuals.
- Calculate the explanatory competence before and after applying the case-based counterfactual generation technique to measure improvement.

### **Experiments and Results**

- **Datasets**: Various popular datasets are used to test the approach.
- **Metrics**: The key metrics are the explanatory competence (fraction of queries that can be explained) and the counterfactual distance (how different the counterfactual is from the query).
- **Findings**: The technique significantly improves the explanatory competence and often produces counterfactuals that are closer to the query than baseline methods.

### **Advantages of the Approach**

1. **Sparsity**: Ensures minimal feature changes.
2. **Plausibility**: Generates realistic counterfactuals based on existing data points.
3. **Efficiency**: Utilizes a structured process to find and adapt counterfactuals.

### **Conclusion**

The proposed case-based method effectively leverages existing data patterns to generate meaningful and actionable counterfactual explanations, enhancing the transparency and interpretability of AI systems.

By reusing patterns of good counterfactuals from a case-base, the approach ensures that generated counterfactuals are not only sparse and plausible but also contextually relevant, addressing key challenges in the field of Explainable AI.

---

## Neural Model and Structural Details

---

## Counterfactual Concept