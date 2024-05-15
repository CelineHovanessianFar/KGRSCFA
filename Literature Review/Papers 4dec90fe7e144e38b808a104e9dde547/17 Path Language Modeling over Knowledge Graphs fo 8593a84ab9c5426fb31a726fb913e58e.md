# 17: Path Language Modeling over Knowledge Graphs for Explainable Recommendation

Features:

Questions:

---

The authors of the paper developed PLM-Rec, a recommendation system that uses a path language model to dynamically generate and extend paths within a knowledge graph. This model predicts sequences of entities and relationships starting from a user node, effectively overcoming the limitations of pre-existing graph connections and enabling the recommendation of items not reachable by traditional fixed-path methods. This approach integrates the generation of recommendations and their explanations, making the system both effective and transparent.

---

## Genearal Summary

The paper introduces a novel framework called Path Language Modeling Recommendation (PLM-Rec) that integrates path language modeling with recommendation systems using knowledge graphs (KGs). Traditional KG-based recommender systems typically suffer from recall bias, meaning some items cannot be recommended because they are unreachable within the existing KG structure using a limited number of hops. PLM-Rec addresses this limitation by leveraging a language model trained on paths within the KG to predict and extend paths dynamically, thus enabling recommendations beyond the pre-existing connections in the KG.

This approach not only enhances the reach of the recommendation system to include previously unreachable items but also integrates the generation of recommendations and explanations into a single process. This is significant because it improves the transparency and explainability of recommendations, providing users with understandable paths that explain why specific items were recommended based on their historical interactions encoded in the KG.

The paper validates the effectiveness of PLM-Rec through extensive experiments on multiple real-world e-commerce datasets, showing that it outperforms several state-of-the-art methods in terms of recommendation accuracy and the ability to mitigate recall bias.

---

## Technical Aspects

The technical solution proposed in the paper involves several key components that work together to enable the Path Language Modeling Recommendation (PLM-Rec) system:

1. **Knowledge Graph (KG) Usage**:
    - The KG is used to represent comprehensive information about users, items, and their attributes. Each node in the KG represents an entity (e.g., a user or a product), and edges between nodes represent relationships or interactions (e.g., a user purchased a product).
2. **Path Language Model**:
    - The core innovation is the development of a path language model that predicts possible paths through the KG. This model is trained on sequences of entities and relationships (edges) derived from the KG, similar to how words are used in sentences in natural language processing.
    - The model functions autoregressively, predicting the next entity or relation in a path given the previous entities and relations. This approach allows for dynamic generation of paths that may not exist in the current structure of the KG, overcoming the limitations of fixed, pre-existing paths.
3. **Model Training**:
    - To train the path language model, paths are first extracted using random walks on the KG, starting from a user entity and ending at an item entity. These paths are used to create training sequences.
    - Sequence augmentation techniques are employed to increase the diversity of training data. This involves substituting entities within paths with semantically similar entities, identified using metrics like cosine similarity from embeddings (e.g., Sentence-BERT).
4. **Embedding and Token Representation**:
    - The model employs embeddings for entities and relations, which are vectors that capture the semantic meaning of each entity and relation in the KG. Positional embeddings and type embeddings are also used to maintain the order and type of tokens (entities and relations) in the path sequences.
5. **Decoding and Recommendation Generation**:
    - Using the trained model, paths are decoded from a starting user node until an end item node (or a special end-of-sequence token) is reached. This is done using a strategy like nucleus sampling, which involves sampling the next token from a subset of likely tokens to ensure diversity and relevance in the generated paths.
    - The decoded paths are evaluated for their likelihood, and the items at the end of the most likely paths are recommended to the user.
6. **Integration of Explanation**:
    - Since the recommendations are directly derived from the decoded paths, the paths themselves serve as natural explanations for why items were recommended. These paths illustrate the user's historical or inferred interests according to the KG.

By integrating these components, the PLM-Rec framework not only provides recommendations but also aligns the recommendations with intuitive, path-based explanations, improving the transparency of the system and potentially increasing user trust and satisfaction.

---

## Neural Model and Structural Details

---

## Counterfactual Concept