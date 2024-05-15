# 26: Counterfactual Reasoning with Knowledge Graph Embeddings

Features:

Questions:

---

---

## Genearal Summary

This paper introduces a new task called CounterFactual Knowledge Graph Reasoning (CFKGR), which links knowledge graph completion (KGC) with counterfactual reasoning. Here’s a detailed summary of the paper's main points:

1. **Background and Motivation**:
    - **Knowledge Graph Embeddings (KGEs)**: These were initially designed to infer true but missing facts in incomplete knowledge graphs (KGs).
    - **Counterfactual Reasoning**: This involves reasoning about hypothetical scenarios and their effects on the current state of the world. It is a key aspect of human cognition and is increasingly relevant for AI systems.
2. **CFKGR Task**:
    - The CFKGR task models the original world state as a knowledge graph. Hypothetical scenarios are represented as edges added to the graph, and plausible changes are inferred from logical rules.
    - The goal is to classify the validity of facts given these hypothetical scenarios.
3. **Dataset Creation**:
    - The authors created benchmark datasets for CFKGR, containing diverse hypothetical scenarios and plausible changes to the original knowledge graph.
    - These datasets are partially human-verified to ensure the plausibility of the changes.
4. **COULDD Method**:
    - The paper introduces COULDD (COUnterfactual Reasoning with KnowLedge Graph EmbeDDings), a method for adapting existing KGEs to accommodate hypothetical scenarios.
    - COULDD updates the embeddings with counterfactual information and stops training once the hypothetical scenario is classified as valid.
5. **Evaluation**:
    - The authors evaluate COULDD on their benchmark datasets and compare its performance with that of pre-trained KGEs and ChatGPT.
    - Results indicate that COULDD improves the ability of KGEs to detect plausible counterfactual changes while maintaining performance on unaffected triples.
    - ChatGPT outperforms KGEs in detecting plausible changes but struggles with knowledge retention.
6. **Human Annotation**:
    - The data generation process and underlying assumptions are validated through human annotation. The annotators assessed the plausibility of the generated scenarios and changes.
7. **Results and Findings**:
    - KGEs adapted with COULDD show solid performance in detecting counterfactual changes.
    - ChatGPT performs better at detecting plausible additions but has poor knowledge retention, leading to mixed results when compared to human annotations.
8. **Contributions**:
    - The paper links KGC and counterfactual reasoning, introducing a challenging task for counterfactual reasoning on KGs.
    - It provides datasets and a new method (COULDD) for adapting KGEs to hypothetical scenarios.
    - The comparison with ChatGPT highlights strengths and weaknesses in current AI systems' counterfactual reasoning capabilities.

In summary, the paper presents a novel approach to combine KGC with counterfactual reasoning, provides datasets for benchmarking, and introduces COULDD to adapt KGEs for counterfactual scenarios, with comprehensive evaluations against both traditional KGEs and modern AI systems like ChatGPT.

---

## Technical Aspects

Here’s an explanation of the technical aspects of their solution in more detail:

### **CFKGR Task Definition**

1. **Knowledge Graph Structure**:
    - The knowledge graph *G*={*E*,*R*,*F*} consists of entities *E*, relations *R*, and facts *F*, where each fact is represented as a triple (*h*,*r*,*t*) indicating that a relation *r* holds between head entity *h* and tail entity *t*.
2. **Hypothetical Scenario**:
    - A hypothetical scenario is represented by a triple *τc*=(*h*,*r*,*t*) which is not originally present in *F*.
    - The counterfactual graph *Gc* is defined as the original graph *G* with the addition of new facts *F*+ induced by the hypothetical scenario and the removal of facts *F*− that contradict the scenario.

### **Dataset Creation**

1. **Rule Mining**:
    - Rules of the form (*X*,*r*1,*Y*)∧(*Y*,*r*2,*Z*)→(*X*,*r*3,*Z*) are mined from the knowledge graph to capture plausible patterns.
        
        (𝑋,𝑟1,𝑌)∧(𝑌,𝑟2,𝑍)→(𝑋,𝑟3,𝑍)
        
    - These rules help generate hypothetical scenarios and their implications.
2. **Generating Scenarios and Inferences**:
    - For each rule, the hypothetical scenarios are created by altering facts to trigger the rule.
    - Plausible additions *F*+ are inferred from these rules, and random corruptions are created for negative sampling.
        
        𝐹+
        

### **COULDD Method**

1. **Initialization**:
    - COULDD starts with pre-trained KGEs, which have embeddings for entities and relations trained on the original knowledge graph.
2. **Training with Hypothetical Scenarios**:
    - For each hypothetical scenario *τc*, the model fine-tunes the embeddings with a batch consisting of *τc* and additional randomly sampled edges from the training graph.
    - Negative samples are generated by corrupting the head or tail of the triples in the batch.
3. **Objective Function**:
    - The standard cross-entropy loss is used to update the embeddings.
    - Training stops once the hypothetical scenario *τc* is classified as valid according to the relation-specific threshold *μr*.
4. **Inference**:
    - After updating the embeddings, the model scores the test cases associated with the hypothetical scenario to determine their plausibility.

### **Evaluation Metrics**

1. **F1 Score**:
    - Measures the overall predictive performance on counterfactual graphs, considering both precision and recall.
2. **Accuracy on Changed Facts**:
    - Evaluates the model’s ability to predict new facts that emerge due to the hypothetical scenario.
3. **F1 Score on Unchanged Facts**:
    - Assesses the model’s ability to retain knowledge of facts that remain unaffected by the hypothetical scenario.

### **Results and Findings**

1. **Comparison with Baseline KGEs**:
    - COULDD improves the detection of plausible counterfactual changes compared to pre-trained KGEs while maintaining performance on unchanged facts.
2. **Comparison with ChatGPT**:
    - ChatGPT performs better in detecting plausible changes but struggles with knowledge retention, highlighting different strengths and weaknesses compared to KGEs.

### **Technical Summary**

1. **KGEs**:
    - Various KGE methods like TransE, ComplEx, RESCAL, ConvE, and TuckER are used, each employing different scoring functions and optimization techniques.
2. **Training and Fine-Tuning**:
    - KGEs are fine-tuned with additional samples and hypothetical scenarios using the COULDD method, leveraging cross-entropy loss and stopping criteria based on scenario validity.
3. **Handling Hypothetical Scenarios**:
    - The approach effectively adapts to hypothetical scenarios by updating embeddings, ensuring plausible inferences while retaining unaffected knowledge.

By leveraging mined rules and fine-tuning KGEs with hypothetical scenarios, COULDD demonstrates an improved capability to reason about counterfactual changes in knowledge graphs.

---

## Neural Model and Structural Details

---

## Counterfactual Concept

**Creating Hypothetical Scenarios**:

- For each mined rule, hypothetical scenarios are generated by identifying existing triples that match the body of the rule and altering one of the body triples to form a new counterfactual triple *τc*.