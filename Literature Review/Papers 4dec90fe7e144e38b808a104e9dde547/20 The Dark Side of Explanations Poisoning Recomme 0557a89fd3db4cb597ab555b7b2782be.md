# 20: The Dark Side of Explanations: Poisoning Recommender Systems with Counterfactual Examples

Features:

Questions:

---

The authors developed H-CARS, a method to poison recommender systems by training a surrogate model on counterfactual explanations to understand the system's logic. They then crafted fake user profiles and interactions to manipulate item embeddings and deceive the system into recommending specific target items more frequently to legitimate users. This approach successfully demonstrated how explainability features, specifically counterfactual explanations, could be exploited to undermine system security.

---

## Genearal Summary

The paper "The Dark Side of Explanations: Poisoning Recommender Systems with Counterfactual Examples" addresses the potential security vulnerabilities in recommender systems caused by counterfactual explanations (CFs). While CFs are generally used to increase transparency and explainability in recommender systems by indicating how item recommendations could change if certain user-item interactions were different, this paper explores how they could be maliciously exploited.

The authors introduce a novel attack strategy, named H-CARS (Horn-Clause Attacks to Recommender Systems), that uses CFs to conduct poisoning attacks on these systems. The method involves training a surrogate model using logical reasoning on CFs, then crafting fake user profiles and interactions to deceive the target recommender system. The attackers manipulate the system by creating optimal item embeddings and fake interactions that increase the likelihood of specific target items being recommended to legitimate users.

This poisoning approach is evaluated through experiments on real datasets (Yelp and MovieLens), demonstrating significant success in manipulating the recommendations. The results underscore the potential risks of CFs in security-sensitive environments, prompting a need for more robust defense mechanisms in explainable AI systems, particularly those using CFs. The paper concludes by highlighting the importance of considering security implications when deploying explainability methods in recommender systems and suggests future research directions to strengthen system defenses against such attacks.

---

## Technical Aspects

The paper "The Dark Side of Explanations: Poisoning Recommender Systems with Counterfactual Examples" introduces a novel approach, H-CARS (Horn-Clause Attacks to Recommender Systems), for poisoning recommender systems by leveraging counterfactual explanations (CFs). Here's a breakdown of the key technical aspects of their solution:

### **1. Surrogate Model Training**

- **Logical Reasoning Model:** The attackers first train a surrogate model that mimics the decision-making process of the target recommender system. This model is based on logical reasoning, specifically using a framework called Neural Collaborative Reasoning (NCR). NCR integrates logical reasoning into a neural network architecture, encapsulating user-item interactions as Horn clauses.
- **Horn Clauses:** In logical terms, a Horn clause can be understood as an implication between antecedents (a conjunction of literals) and a consequent (a single literal). In the context of recommender systems, this structure allows the model to assess whether a user will like an item based on their interaction history.

### **2. Counterfactual Explanations**

- **Generation and Utilization:** Counterfactual explanations are generated to determine minimal changes in a user’s interaction history that would result in a different recommendation outcome. These CFs are used to augment the training data for the surrogate model, helping it learn the underlying decision boundaries and logic of the target system more effectively.

### **3. Poisoning Attack via Fake Interactions**

- **Optimization Framework:** Once the surrogate model is trained, the attackers use it to compute optimal item embeddings that can be manipulated to favor certain target items. This involves reversing the traditional optimization approach; instead of directly retraining the recommender model, they calculate the best way to alter item embeddings to manipulate recommendations.
- **Fake User Profiles:** Using the surrogate model, attackers craft fake user profiles and interaction records. These profiles are designed to interact with the system in a way that the manipulated item embeddings cause the target items to be recommended more frequently to legitimate users.

### **4. Evaluating Attack Impact**

- **Metrics:** The effectiveness of the attack is measured by the hit ratio (HR), which assesses the probability that manipulated target items appear in the top recommended items for a real user.
- **Experimental Validation:** The method is tested on two real-world datasets (Yelp and MovieLens), demonstrating its ability to effectively manipulate recommendation outcomes.

### **5. Data Augmentation with CFs**

- The use of CFs not only helps in enhancing the performance of the surrogate model by providing additional training data but also addresses the issue of spurious correlations by contrasting differences between factual and counterfactual interaction patterns.

This approach showcases a sophisticated use of counterfactual reasoning to exploit the vulnerabilities of recommender systems, highlighting significant security implications, especially for systems where transparency and explainability are critical.

---

## Neural Model and Structural Details

---

## Counterfactual Concept