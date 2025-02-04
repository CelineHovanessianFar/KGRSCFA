import networkx as nx
from community import community_louvain
import collections
from CAFE import utils
import numpy as np
import os
import pickle
from kneed import KneeLocator
from data_registry import DataRegistry

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
        self.G = G
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
            for entity_id, edges in DataRegistry.kg.G.get(entity, {}).items():
                relation = DataRegistry.kg.relation_info(entity, 'product')
                targets = edges.get(relation, [])
                for target in targets:
                    # Add edge between the word and the product it describes
                    entity_edges.append(((entity, entity_id), ('product', target), {'relation': relation}))
            
            # Add edges to the graph
            G.add_edges_from(entity_edges)
            
            # Check for connections between products and other words
            product_edges = []
            for product_id, product_edges_dict in DataRegistry.kg.G.get('product', {}).items():
                relation = DataRegistry.kg.relation_info('product', entity)
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

    def adjusted_degree_centrality(self, number_of_products):
        # TODO
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
            number_of_non_product_nodes = sum(1 for n in self.G.nodes if not n.startswith('product'))
            
            # Initialize the centrality dictionary
            centrality = {}


            # Apply different normalization based on node type
            for node, degree in raw_centrality:
                if node.startswith('product'):
                    # Normalize by the number of non-product nodes
                    centrality[node] = degree / number_of_non_product_nodes
                else:
                    # Normalize by the number of products
                    centrality[node] = degree / number_of_products

            # Save the centrality data to the file
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(centrality, f)

        return centrality


class CommunityAnalyzer:
    """
    A class for analyzing communities within a knowledge graph (KG).

    This class provides functionality for:
    - Calculating centrality measures to evaluate the importance of nodes in the graph.
    - Identifying communities using various centrality-based outlier removal methods.
    - Reporting the composition of identified communities by node type.

    Attributes:
        kg_obj (object): The loaded knowledge graph object.
        kg (dict): The raw knowledge graph data.
        G (nx.Graph): The graph representation of the KG.
        centrality_dict (dict): Adjusted degree centrality scores for the nodes.
        degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality (dict): 
            Centrality measures, initialized as None.
        report (dict): A report summarizing community composition.
        partition (dict): A mapping of nodes to their respective communities.

    Methods:
        build_graph(kg):
            Constructs a networkx graph from the knowledge graph.

        identify_communities(method='top_n', top_n=10, num_std_dev=2, top_percentile=95):
            Detects communities using the Louvain method, optionally removing centrality outliers.

        report_community_composition():
            Generates a summary of the community composition by node type.
    """
    def __init__(self):
        self.G = self.build_graph(DataRegistry.kg.G)
        CS = CentralityScores(self.G)
        self.centrality_dict = CS.adjusted_degree_centrality(number_of_products=len(DataRegistry.kg.G['product']))
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


    def identify_communities(self, method='knee', top_n=10, num_std_dev=2, top_percentile=95):
        """
        Identifies communities in the graph after removing outlier nodes based on centrality.

        This method uses various statistical and heuristic techniques to identify and remove 
        outlier nodes in the graph based on their centrality values. The cleaned graph is 
        then used for community detection.

        Parameters:
        ----------
        method : str, optional
            The method to use for identifying and removing centrality outliers. Options include:
            - 'std_dev': Remove nodes with centrality values beyond a specified number of 
            standard deviations from the mean.
            - 'top_n': Remove the top `n` nodes with the highest centrality values.
            - 'iqr': Remove nodes with centrality values falling outside the interquartile range (IQR).
            - 'percentile': Remove nodes whose centrality values are above the `top_percentile`.
            - 'knee': Remove nodes above the "knee" point in the centrality value distribution curve.
            - None: Retain all nodes (no outlier removal).
            
            Default is 'top_n'.

        top_n : int, optional
            The number of nodes with the highest centrality values to remove when using the 
            'top_n' method. Default is 10.

        num_std_dev : int or float, optional
            The number of standard deviations from the mean to use as thresholds for removing 
            outliers in the 'std_dev' method. Default is 2.

        top_percentile : int, optional
            The percentile threshold for removing nodes with centrality values above this 
            percentile when using the 'percentile' method. Default is 95.

        Returns:
        -------
        networkx.Graph
            A new graph with the identified outlier nodes removed. If no method is specified 
            (`method=None`), the original graph is returned.

        Raises:
        ------
        ValueError
            If an invalid method is specified.

        Notes:
        ------
        - The method uses `self.centrality_dict` to access the centrality values of the nodes 
        and `self.G` as the base graph.
        - Removing nodes also removes their associated edges in the resulting graph.
        """
        
        def remove_centrality_outliers(method='std_dev', **kwargs):
            outlier_checker = CentralityOutlierChecker(method=method, thresholds=kwargs)
            nodes_to_remove = [
                node for node, centrality in self.centrality_dict.items()
                if outlier_checker.is_outlier(centrality, method=method)
            ]

            G_new = self.G.copy()
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
    """
    A class to identify outliers in centrality measures using various statistical methods.

    This class provides functionality to load centrality measures from a file, compute bounds for 
    identifying outliers based on different statistical methods, and check if a given centrality 
    value is an outlier.

    Parameters:
    -----------
    method : str, optional
        The default method used for outlier detection. Options are 'std_dev', 'top_n', 'percentile', 'iqr', 
        and 'knee'. Default is 'knee'.
    thresholds : dict, optional
        A dictionary of thresholds used for different methods. Default values are:
        - 'std_dev': 2 (standard deviations)
        - 'top_n': 20 (top N values)
        - 'percentile': 95 (percentile)
        - 'iqr': 1.5 (interquartile range)

    Attributes:
    -----------
    method : str
        The default method used for outlier detection.
    thresholds : dict
        A dictionary of thresholds for outlier detection methods.
    centrality_bounds : dict
        A dictionary storing the calculated bounds for each outlier detection method.
    centrality_dict : dict
        A dictionary containing centrality measures loaded from a file.

    Methods:
    --------
    load_centrality_dict():
        Loads the centrality measures from a pickle file.

    compute_all_centrality_bounds():
        Computes the bounds for each outlier detection method based on the centrality measures.

    is_outlier(centrality, method=None):
        Checks if a given centrality value is an outlier based on the specified method.

    Raises:
    -------
    FileNotFoundError:
        If the centrality dictionary file is not found.
    ValueError:
        If an unsupported method is specified for outlier detection.
    """

    def __init__(self, method='knee', thresholds=None):
        self.method = method
        default_thresholds = {
            'std_dev': 2,
            'top_n': 20,
            'percentile': 95,
            'iqr': 1.5  # Adding default for IQR as well
        }
        
        self.thresholds = {**default_thresholds, **(thresholds or {})}
        self.centrality_dict = self.load_centrality_dict()
        self.centrality_bounds = self.compute_all_centrality_bounds()

    def load_centrality_dict(self):
        """
        Load the centrality measures from a pickle file.

        Returns:
        --------
        dict
            A dictionary containing centrality measures.

        Raises:
        -------
        FileNotFoundError:
            If the centrality dictionary file is not found.
        """
        file_path = 'tmp/multimodal_degree_centralities.pkl'
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError("Centrality dict not found")

    def compute_all_centrality_bounds(self):
        """
        Compute the bounds for each outlier detection method based on the centrality measures.

        This method calculates the bounds for the following methods:
        - Standard deviation ('std_dev')
        - Top N values ('top_n')
        - Interquartile range ('iqr')
        - Percentile ('percentile')
        - Knee point ('knee')
        """
        centrality_values = list(self.centrality_dict.values())
        sorted_centralities = sorted(centrality_values, reverse=True)

        mean = np.mean(centrality_values)
        std = np.std(centrality_values)
        bounds = {}

        # Standard Deviation Method
        bounds['std_dev'] = (
            mean - self.thresholds['std_dev'] * std, 
            mean + self.thresholds['std_dev'] * std
        )

        # Top N Values Method
        bounds['top_n'] = (
            None,
            sorted_centralities[min(self.thresholds['top_n'] - 1, len(sorted_centralities) - 1)]
        )

        # Interquartile Range (IQR) Method
        Q1, Q3 = np.percentile(centrality_values, [25, 75])
        IQR = Q3 - Q1
        bounds['iqr'] = (
            Q1 - self.thresholds['iqr'] * IQR, 
            Q3 + self.thresholds['iqr'] * IQR
        )

        # Percentile Method
        bounds['percentile'] = (
            None, 
            np.percentile(centrality_values, self.thresholds['percentile'])
        )

        # Knee Point Method
        x = np.arange(0, len(centrality_values))
        knee_locator = KneeLocator(x, sorted_centralities, curve='convex', direction='decreasing')
        knee_value = sorted_centralities[knee_locator.knee] if knee_locator.knee is not None else None
        bounds['knee'] = (None, knee_value)

        return bounds

    def compute_centrality_bound(self, threshold, method):
        """
        Compute the bound for a specific outlier detection method based on the centrality measures.

        Parameters:
            threshold (float): The threshold to use for the selected method.
            method (str): The outlier detection method to calculate the bound for.
                        Options: 'std_dev', 'top_n', 'iqr', 'percentile', 'knee'

        Returns:
            tuple: The computed bound for the selected method.
        """
        centrality_values = list(self.centrality_dict.values())
        sorted_centralities = sorted(centrality_values, reverse=True)

        if method == 'std_dev':
            mean = np.mean(centrality_values)
            std = np.std(centrality_values)
            return (
                mean - threshold * std,
                mean + threshold * std
            )

        elif method == 'top_n':
            return (
                None,
                sorted_centralities[min(int(threshold) - 1, len(sorted_centralities) - 1)]
            )

        elif method == 'iqr':
            Q1, Q3 = np.percentile(centrality_values, [25, 75])
            IQR = Q3 - Q1
            return (
                Q1 - threshold * IQR,
                Q3 + threshold * IQR
            )

        elif method == 'percentile':
            return (
                None,
                np.percentile(centrality_values, threshold)
            )

        elif method == 'knee':
            x = np.arange(0, len(centrality_values))
            knee_locator = KneeLocator(x, sorted_centralities, curve='convex', direction='decreasing')
            knee_value = sorted_centralities[knee_locator.knee] if knee_locator.knee is not None else None
            return (None, knee_value)

        else:
            raise ValueError(f"Unsupported method: {method}")

    def is_outlier(self, centrality, method=None, threshold=None):
        """
        Check if a given centrality value is an outlier based on the specified method.

        Parameters:
        -----------
        centrality : float
            The centrality value to check.
        method : str, optional
            The method to use for outlier detection. If None, the default method is used.

        Returns:
        --------
        bool
            True if the centrality value is an outlier, False otherwise.

        Raises:
        -------
        ValueError:
            If an unsupported method is specified for outlier detection.
        """
        if method is None:
            method = self.method
        
        if method not in self.centrality_bounds:
            raise ValueError(f"Method {method} not supported.")

        if threshold is not None:
            centrality_bounds = self.compute_centrality_bound(threshold, method)
        else:
            centrality_bounds = self.centrality_bounds[method]
            
        # lower_bound, upper_bound = centrality_bounds[method]
        lower_bound, upper_bound = centrality_bounds
        if method in ['std_dev', 'iqr']:
            return centrality < lower_bound or centrality > upper_bound
        elif method in ['top_n', 'knee']:
            return centrality > upper_bound if upper_bound is not None else False
        elif method == 'percentile':
            return centrality >= upper_bound

        return False
