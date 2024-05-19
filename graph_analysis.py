import networkx as nx
from community import community_louvain

def calculate_and_identify_top_centralities(G):
    # Calculate centralities
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)  # Increase max_iter and tol
    except nx.PowerIterationFailedConvergence:
        # Fallback to using the `arpack` method if the power method fails
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06, method='arpack')
    
    # Sort nodes by centrality
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'degree_centrality': top_degree,
        'betweenness_centrality': top_betweenness,
        'closeness_centrality': top_closeness,
        'eigenvector_centrality': top_eigenvector,
    }

class CentralityScores:
    def __init__(self) -> None:
        self.kg_obj = utils.load_kg('beauty')
        self.kg = self.kg_obj.G
        self.relation_info = self.kg_obj.relation_info
        self.category_product_subgraph = None
        self.brand_product_subgraph = None
        self.related_product_subgraph = None
        self.word_product_subgraph = None

    # Generic
    def extract_product_category_subgraph(self):
        G = nx.Graph()
        product_related_entities = ['brand', 'category', 'related_product', 'word']
        for entity in product_related_entities:
            # Collect all edges involving word nodes
            entity_edges = []
            
            # Iterate over each word and its connections
            for entity_id, edges in self.kg.get(entity, {}).items():
                relation = self.relation_info(entity, 'product')
                targets = edges.get(relation, [])
                for target in targets:
                    # Add edge between the word and the product it describes
                    entity_edges.append(((entity, entity_id), ('product', target), {'relation': relation}))
            
            # Add edges to the graph
            G.add_edges_from(entity_edges)
            
            # Check for connections between products and other words
            product_edges = []
            for product_id, product_edges_dict in self.kg.get('product', {}).items():
                relation = self.relation_info('product', entity)
                product_targets = product_edges_dict.get(relation, [])
                for target in product_targets:
                    product_edges.append((('product', product_id), (entity, target), {'relation': relation}))
            
            # Add product to word edges
            G.add_edges_from(product_edges)
            setattr(self, f'{entity}_product_subgraph', G)

    def subgraphs_centrality_scores(self):
        product_related_entities = ['brand', 'category', 'related_product', 'word']
        centrality_scores = {}
        
        for entity in product_related_entities:
            subgraph = getattr(self, f'{entity}_product_subgraph')
            if subgraph is not None:
                degree_centrality = nx.degree_centrality(subgraph)
                betweenness_centrality = nx.betweenness_centrality(subgraph)
                closeness_centrality = nx.closeness_centrality(subgraph)
                eigenvector_centrality = nx.eigenvector_centrality(subgraph)
                
                centrality_scores[entity] = {
                    'degree_centrality': degree_centrality,
                    'betweenness_centrality': betweenness_centrality,
                    'closeness_centrality': closeness_centrality,
                    'eigenvector_centrality': eigenvector_centrality
                }

        self.centrality_scores = centrality_scores


class CommunityAnalyzer:
    def __init__(self, kg):
        self.kg = kg
        self.G = self.build_graph(kg)
        self.degree_centrality = None
        self.betweenness_centrality = None
        self.closeness_centrality = None
        self.eigenvector_centrality = None
        self.report = None
        self.partition = None

    def build_graph(self, kg):
        G = nx.Graph()
        
        for product_id, attributes in kg['product'].items():
            product_node = f"product_{product_id}"
            G.add_node(product_node, type='product')
            
            # Add category nodes and edges
            for category_id in attributes.get('belongs_to', []):
                category_node = f"category_{category_id}"
                G.add_node(category_node, type='category')
                G.add_edge(product_node, category_node, relationship='belongs_to')
                
            # Add brand nodes and edges
            for brand_id in attributes.get('produced_by', []):
                brand_node = f"brand_{brand_id}"
                G.add_node(brand_node, type='brand')
                G.add_edge(product_node, brand_node, relationship='produced_by')
                
            # Add word nodes and edges
            for word_id in attributes.get('described_by', []):
                word_node = f"word_{word_id}"
                G.add_node(word_node, type='word')
                G.add_edge(product_node, word_node, relationship='described_by')
                
            # Add related product nodes and edges
            for related_id in attributes.get('also_bought', []):
                related_product_node = f"related_product_{related_id}"
                G.add_node(related_product_node, type='related_product')
                G.add_edge(product_node, related_product_node, relationship='also_bought')
                
            for related_id in attributes.get('also_viewed', []):
                related_product_node = f"related_product_{related_id}"
                G.add_node(related_product_node, type='related_product')
                G.add_edge(product_node, related_product_node, relationship='also_viewed')
                
            for related_id in attributes.get('bought_together', []):
                related_product_node = f"related_product_{related_id}"
                G.add_node(related_product_node, type='related_product')
                G.add_edge(product_node, related_product_node, relationship='bought_together')
                
        return G

    def identify_communities(self):
        self.partition = community_louvain.best_partition(self.G)
        # Create a new partition dictionary with node types included
        partition_with_types = {}
        for node, community in self.partition.items():
            node_type = self.G.nodes[node].get('type', 'unknown')
            partition_with_types[(node, node_type)] = community
        return partition_with_types
