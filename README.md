# Counterfactual Analysis for Path-Based Knowledge-Graph Recommender Systems

## Introduction
This study focuses on developing and applying a counterfactual analysis for path-based knowledge-graph recommender systems. The primary objective is to expand the explainability of the recommendations by using counterfactual scenarios, through the selection and evaluation of relevant attributes and behavioral scenarios, specially for the provider side.

## Methodology
The developed framework takes in a recommended path and explains how the recommendation might change if the product had different hypothetical attributes. For a given recommended product, the framework examines a set of hypothetical attributes and interactions. This involves extracting all attributes and interactions for the recommended product and identifying all products linked to those entities. The attributes of these connected products were assumed to be relevant for the counterfactual analysis. Trying all possible attributes is challenging due to the large search space, which underscores the value of this targeted approach.

To enhance the counterfactual process, I performed community identification and degree centrality analysis on the knowledge graph. Among the connected attributes, if the number of selected attributes for each type exceeded a predefined threshold, I prioritized the attributes within the same community as the selected product.

The analysis is based on the assumption that the system understands the user's preferences based on their purchased products. To test if a product with different attributes would still be recommended, I conducted an isolated analysis of each attribute. This involved taking the corresponding metapath for that attribute within the system and calculating the recommendation score. If the score of the path connecting the user to the product (through any of the user's purchased products) was higher than the score of the 10th recommended product, it suggested that the product would still be recommended with the different attribute.

## Examples
An example would be testing counterfactual interactions such as viewing a certain other product leading to viewing the recommended product, or if a product is cruelty-free, to see if that scenario is still plausible. This provides extended explainability for the provider side.

## Expected Outcomes
The expected outcomes of this analysis include identifying attributes and interactions that, if present, would lead to the selection of the product. This could increase user satisfaction, enhance market analysis, and contribute to greater recommendation diversity.

## Conclusion
This structured approach allows for a comprehensive evaluation of the counterfactual scenarios, demonstrating the potential impacts of varying product attributes on recommendation outcomes.


## To Do
- [ ] Code Refactor
- [ ] Report Completion

## Already Done
- [x] Proposal
- [x] Literature Review Summaries
- [x] Literature Chapter Draft
- [x] Methodology Chapter Draft
- [x] Code Completed