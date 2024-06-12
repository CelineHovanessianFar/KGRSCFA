import networkx as nx
from community import community_louvain
import collections
from CAFE import utils
import numpy as np
import os
import pickle
from kneed import KneeLocator

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
    def __init__(self, G) -> None:
        self.kg_obj = utils.load_kg('beauty')
        self.G = G
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

    def adjusted_degree_centrality(self, number_of_products=859):
        file_path = 'tmp/multimodal_degree_centralities.pkl'
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Load the centrality data from the file
            with open(file_path, 'rb') as f:
                centrality = pickle.load(f)
        else:
            # Calculate raw degree centrality for all nodes
            raw_centrality = self.G.degree()

            # Count the total number of nodes except 'product' type nodes
            non_product_nodes = sum(1 for n in self.G.nodes if not n.startswith('product'))
            
            # Initialize the centrality dictionary
            centrality = {}

            # Apply different normalization based on node type
            for node, degree in raw_centrality:
                if node.startswith('product'):
                    # Normalize by the number of non-product nodes
                    centrality[node] = degree / non_product_nodes
                else:
                    # Normalize by the number of products
                    centrality[node] = degree / number_of_products

            # Save the centrality data to the file
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(centrality, f)

        return centrality


class CommunityAnalyzer:
    def __init__(self):
        self.kg_obj = utils.load_kg('beauty')
        self.kg = self.kg_obj.G
        self.G = self.build_graph(self.kg)
        CS = CentralityScores(self.G)
        self.centrality_dict = CS.adjusted_degree_centrality()
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


    def identify_communities(self, method='top_n', top_n=10, num_std_dev=2, top_percentile=95):
        def remove_centrality_outliers(method='std_dev', num_std_dev=2, top_n=10, top_percentile=95):
            centrality_values = list(self.centrality_dict.values())
            removed_nodes = {}

            if method == 'std_dev':
                mean_centrality = np.mean(centrality_values)
                std_dev_centrality = np.std(centrality_values)
                lower_bound = mean_centrality - num_std_dev * std_dev_centrality
                upper_bound = mean_centrality + num_std_dev * std_dev_centrality
                for node, centrality in self.centrality_dict.items():
                    if centrality < lower_bound or centrality > upper_bound:
                        removed_nodes[node] = centrality
            elif method == 'top_n':
                sorted_centralities = sorted(self.centrality_dict.items(), key=lambda item: item[1], reverse=True)
                # Collect top_n outliers
                for node, centrality in sorted_centralities[:top_n]:
                    removed_nodes[node] = centrality
            elif method == 'iqr':
                Q1 = np.percentile(centrality_values, 25)
                Q3 = np.percentile(centrality_values, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                for node, centrality in self.centrality_dict.items():
                    if centrality < lower_bound or centrality > upper_bound:
                        removed_nodes[node] = centrality
            elif method == 'percentile':
                threshold = np.percentile(centrality_values, top_percentile)
                # Collect percentile-based outliers
                for node, centrality in self.centrality_dict.items():
                    if centrality >= threshold:
                        removed_nodes[node] = centrality
            
            elif method == None:
                return self.G
            else:
                raise ValueError("Invalid method specified. Choose 'std_dev', 'top_n', 'iqr', or 'percentile'.")

            # Create a new graph without outliers
            G_new = self.G.copy()
            nodes_to_remove = set(removed_nodes.keys())
            # TODO: does this remove the edges too? is this consistent?
            G_new.remove_nodes_from(nodes_to_remove)

            return G_new

    # Remove outliers based on centrality and get the cleaned graph and removed nodes
        G_clean = remove_centrality_outliers(method=method, top_n=top_n, num_std_dev=num_std_dev, top_percentile=top_percentile)


        # Community detection using Louvain method
        partition = community_louvain.best_partition(G_clean, random_state=42)
        partition_with_types = {}
        for node, community in partition.items():
            node_type = self.G.nodes[node].get('type', 'unknown')
            partition_with_types[(node, node_type)] = community
        return partition_with_types


    def report_community_composition(self):
        # This method assumes that 'identify_communities' has been called first
        if self.partition is None:
            raise ValueError("Communities not identified. Please run identify_communities() first.")

        # Initialize a dictionary to store the counts
        community_composition = collections.defaultdict(lambda: collections.Counter())
        
        # Iterate over each node and its community
        for node, community in self.partition.items():
            node_type = self.G.nodes[node]['type']  # Get the type of the node
            community_composition[community][node_type] += 1  # Increment the count of the type in its community

        # Create a readable report from the composition data
        report = {}
        for community, types in community_composition.items():
            report[f"Community {community}"] = dict(types)

        return report


class CentralityOutlierChecker:
    def __init__(self, method='std_dev', thresholds=None):
        self.method = method
        self.thresholds = thresholds if thresholds is not None else {
            'std_dev': 2,
            'top_n': 20,
            'percentile': 95,
            'iqr': 1.5  # Adding default for IQR as well
        }
        self.centrality_bounds = {}
        self.centrality_dict = self.load_centrality_dict()
        self.compute_all_centrality_bounds()

    def load_centrality_dict(self):
        file_path = 'tmp/multimodal_degree_centralities.pkl'
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError("Centrality dict not found")

    def compute_all_centrality_bounds(self):
        centrality_values = list(self.centrality_dict.values())
        sorted_centralities = sorted(centrality_values, reverse=True)

        mean = np.mean(centrality_values)
        std = np.std(centrality_values)
        self.centrality_bounds['std_dev'] = (
            mean - self.thresholds['std_dev'] * std, 
            mean + self.thresholds['std_dev'] * std
        )

        self.centrality_bounds['top_n'] = (
            None, 
            sorted_centralities[min(self.thresholds['top_n'] - 1, len(sorted_centralities) - 1)]
        )

        Q1, Q3 = np.percentile(centrality_values, [25, 75])
        IQR = Q3 - Q1
        self.centrality_bounds['iqr'] = (
            Q1 - self.thresholds['iqr'] * IQR, 
            Q3 + self.thresholds['iqr'] * IQR
        )

        self.centrality_bounds['percentile'] = (
            None, 
            np.percentile(centrality_values, self.thresholds['percentile'])
        )

        x = np.arange(0, len(centrality_values))
        knee_locator = KneeLocator(x, sorted_centralities, curve='convex', direction='decreasing')
        knee_value = sorted_centralities[knee_locator.knee] if knee_locator.knee is not None else None
        self.centrality_bounds['knee'] = (None, knee_value)

    def is_outlier(self, centrality, method=None):
        if method is None:
            method = self.method
        
        if method not in self.centrality_bounds:
            raise ValueError(f"Method {method} not supported or bounds not set.")

        lower_bound, upper_bound = self.centrality_bounds[method]
        if method in ['std_dev', 'iqr']:
            return centrality < lower_bound or centrality > upper_bound
        elif method in ['top_n', 'knee']:
            return centrality > upper_bound if upper_bound is not None else False
        elif method == 'percentile':
            return centrality >= upper_bound

        return False
