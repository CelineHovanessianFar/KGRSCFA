import os
from tqdm import tqdm

from CAFE import utils

from kg_info import KGENtitiesRelationsInfo
from graph_analysis import CommunityAnalyzer
from graph_analysis import *
from data_registry import DataRegistry

import sys
sys.path.append('CAFE')


class EntityFilter:
    def __init__(self):
        self.analyzer = CommunityAnalyzer()
        self.connection_stats, _ = DataRegistry.kg_info.calculate_connection_stats()
        self.Node2Partition = self.analyzer.identify_communities()
        self.partition2Node = self.create_partition_to_node_mapping()
        self.recommended_product_category_products = set()
        

    def create_partition_to_node_mapping(self):
        partition2Node = {}
        for node, partition in self.Node2Partition.items():
            if partition not in partition2Node:
                partition2Node[partition] = []
            partition2Node[partition].append(node)
        return partition2Node

    def filter_entities(self, product_id, related_entities, h_entity_type, t_entity_type, h_entity_id, 
                        force_community_filter=True, filter_by_category=True, outlier_methods=None, thresholds=None):
        # Default thresholds if not provided
        default_thresholds = {
            'std_dev': 2,
            'top_n': 20,
            'percentile': 95,
            'iqr': 1.5
        }
        thresholds = thresholds if thresholds is not None else default_thresholds
        outlier_methods = outlier_methods if outlier_methods is not None else ['knee']

        # Helper function to check outliers using multiple methods and thresholds
        def is_outlier_with_methods(centrality_value):
            return any(
                self.outlier_checker.is_outlier(
                    centrality_value, 
                    method=method, 
                    threshold=thresholds.get(method)
                )
                for method in outlier_methods
            )

        # Centrality check
        centrality_key = f'{h_entity_type}_{h_entity_id}'
        centrality = self.outlier_checker.centrality_dict.get(centrality_key)
        if centrality is None:
            print(f"Warning: Centrality value not found for key: {centrality_key}")
            community_flag = False
        else:
            community_flag = is_outlier_with_methods(centrality)

        centrality = lambda entity: self.outlier_checker.centrality_dict[f'{t_entity_type}_{entity}']

        if community_flag or force_community_filter:
            recommended_product_community = self.Node2Partition.get(('product_{}'.format(product_id), 'product'), None)
            if recommended_product_community is None:
                return set()  # Return an empty set if the product community is not found

            # Only picks ids of entities with the same type as tail entity type
            partition_et_ids = [
                int(node.split('_')[-1])
                for node, type in self.partition2Node.get(recommended_product_community, [])
                if type == t_entity_type
            ]

            if filter_by_category:
                return {entity for entity in related_entities if entity in partition_et_ids and 
                        entity in self.recommended_product_category_products and 
                        not is_outlier_with_methods(centrality(entity))}

        if filter_by_category:
            return {entity for entity in related_entities if entity in self.recommended_product_category_products and 
                    not is_outlier_with_methods(centrality(entity))}
        return {entity for entity in related_entities if not is_outlier_with_methods(centrality(entity))}


    # After collecting the connected products, this method collects attributes related to that products, filters them and
    # ppopulates the counterfactual attributes dictionary
    def update_entities(self, related_products, recommended_product_id, entities, force_community_filter, outlier_methods=None, thresholds=None):
        for related_product_id in related_products:
            related_product_info = DataRegistry.kg.G['product'].get(related_product_id, {})
            category_entities = related_product_info.get('belongs_to', [])
            filtered_categories = self.filter_entities(recommended_product_id, category_entities, 'product', 'category', related_product_id, 
                                                       force_community_filter=force_community_filter, filter_by_category=False, outlier_methods=outlier_methods, 
                                                        thresholds=thresholds)
            entities['category'].update(filtered_categories)

            brand_entities = related_product_info.get('produced_by', [])
            filtered_brands = self.filter_entities(recommended_product_id, brand_entities, 'product', 'brand', related_product_id, 
                                                   force_community_filter=force_community_filter, filter_by_category=False, outlier_methods=outlier_methods, 
                                                    thresholds=thresholds)
            entities['brand'].update(filtered_brands)

            word_entities = related_product_info.get('described_by', [])
            filtered_words = self.filter_entities(recommended_product_id, word_entities, 'product', 'word', related_product_id, 
                                                  force_community_filter=force_community_filter, filter_by_category=False, outlier_methods=outlier_methods, 
                                                    thresholds=thresholds)
            entities['word'].update(filtered_words)

            related_related_products = related_product_info.get('also_bought', []) +\
                                        related_product_info.get('also_viewed', []) +\
                                        related_product_info.get('bought_together', [])
            filtered_related_products = self.filter_entities(recommended_product_id, related_related_products, 'product', 'related_product', related_product_id, 
                                                             force_community_filter=force_community_filter, filter_by_category=False, outlier_methods=outlier_methods, 
                                                            thresholds=thresholds)
            entities['related_product'].update(filtered_related_products)

    # First collects all the related products to the recommender product, if they meet the 
    # filtering criteria, then it collects attributes like brand, category, word and related_product
    # of the connected products, to contstruct counterfactual scenario
    def __call__(self, recommended_product_id, force_community_filter=False, filter_by_category=True, 
                 print_report=True, outlier_method='knee', outlier_methods_products=None, thresholds_products=None,
                 outlier_methods_attributes=None, thresholds_attributes=None):
        entities = {'category': set(), 'brand': set(), 'word': set(), 'related_product': set()}
        self.outlier_checker = CentralityOutlierChecker(method=outlier_method)

        tasks = []

        # Categories connected to the recommended product
        if not filter_by_category:
            category_ids = DataRegistry.kg.G['product'].get(recommended_product_id, {}).get('belongs_to', [])
            for category_id in category_ids:
                related_products = DataRegistry.kg.G['category'].get(category_id, {}).get('rev_belongs_to', [])
                related_products = self.filter_entities(recommended_product_id, related_products, 'category', 'product', category_id, 
                                                        force_community_filter=force_community_filter, filter_by_category=filter_by_category, outlier_methods=outlier_methods_products, 
                                                        thresholds=thresholds_products)
                tasks.append(('category', category_id, related_products))
        else: 
            recommended_product_category = set(DataRegistry.kg.G['product'].get(recommended_product_id, {}).get('belongs_to', []))
            for category in recommended_product_category:
                # Fetch products related to each category
                category_products = DataRegistry.kg.G['category'].get(category, {}).get('rev_belongs_to', [])
                self.recommended_product_category_products.update(category_products)

        # Brands connected to the recommended product
        brand_ids = DataRegistry.kg.G['product'].get(recommended_product_id, {}).get('produced_by', [])
        for brand_id in brand_ids:
            related_products = DataRegistry.kg.G['brand'].get(brand_id, {}).get('rev_produced_by', [])
            related_products = self.filter_entities(recommended_product_id, related_products, 'brand', 'product', brand_id, 
                                                    force_community_filter=force_community_filter, filter_by_category=filter_by_category, outlier_methods=outlier_methods_products, 
                                                    thresholds=thresholds_products)
            tasks.append(('brand', brand_id, related_products))

        # Words connected to the recommended product
        word_ids = DataRegistry.kg.G['product'].get(recommended_product_id, {}).get('described_by', [])
        for word_id in word_ids:
            related_products = DataRegistry.kg.G['word'].get(word_id, {}).get('rev_described_by', [])
            related_products = self.filter_entities(recommended_product_id, related_products, 'word', 'product', word_id, 
                                                    force_community_filter=force_community_filter, filter_by_category=filter_by_category, outlier_methods=outlier_methods_products, 
                                                    thresholds=thresholds_products)
            
            tasks.append(('word', word_id, related_products))

        # Related Products connected to the recommended product
        related_products_ids = DataRegistry.kg.G['product'].get(recommended_product_id, {}).get('also_bought', []) +\
                            DataRegistry.kg.G['product'].get(recommended_product_id, {}).get('also_viewed', []) +\
                            DataRegistry.kg.G['product'].get(recommended_product_id, {}).get('bought_together', [])
        related_products_ids = self.filter_entities(recommended_product_id, related_products_ids, 'product', 'related_product', recommended_product_id, 
                                                    force_community_filter=force_community_filter, filter_by_category=filter_by_category, outlier_methods=outlier_methods_products, 
                                                        thresholds=thresholds_products)
        tasks.append(('related_product', recommended_product_id, related_products_ids))
        
        # Process all tasks
        for related_products in tqdm(tasks, desc='Filtering Attributes: '):
            self.update_entities(related_products, recommended_product_id, entities, force_community_filter, outlier_methods=outlier_methods_attributes, 
                                                        thresholds=thresholds_attributes)
        
        entities = {k: list(v) for k, v in entities.items()}
        if print_report:
            self.report_related_entities_statistics(entities)
        
        # TODO
        tmp_directory = 'tmp'
        if not os.path.exists(tmp_directory):
            os.makedirs(tmp_directory)

        return entities


    # Generates and prints a statistical report of related entities for analysis.
    def report_related_entities_statistics(self, entities):
        """
        Generates a report on the statistics of the related entities for a given product.
        """

        print("Related Entities Statistics Report:\n")
        
        # Categories
        num_related_categories = len(entities['category'])
        num_kg_categories = len(DataRegistry.kg.G['category'])
        print(f"Number of related categories: {num_related_categories}")
        print(f"Number of categories in the knowledge graph: {num_kg_categories}")
        print(f"Percentage of related categories: {num_related_categories / num_kg_categories * 100:.2f}%\n")
        
        # Brands
        num_related_brands = len(entities['brand'])
        num_kg_brands = len(DataRegistry.kg.G['brand'])
        print(f"Number of related brands: {num_related_brands}")
        print(f"Number of brands in the knowledge graph: {num_kg_brands}")
        print(f"Percentage of related brands: {num_related_brands / num_kg_brands * 100:.2f}%\n")
        
        # Words
        num_related_words = len(entities['word'])
        num_kg_words = len(DataRegistry.kg.G['word'])
        print(f"Number of related words: {num_related_words}")
        print(f"Number of words in the knowledge graph: {num_kg_words}")
        print(f"Percentage of related words: {num_related_words / num_kg_words * 100:.2f}%\n")
        
        # Related Products
        num_related_products = len(entities['related_product'])
        num_kg_related_products = len(DataRegistry.kg.G['related_product'])
        print(f"Number of related products: {num_related_products}")
        print(f"Number of products in the knowledge graph: {num_kg_related_products}")
        print(f"Percentage of related products: {num_related_products / num_kg_related_products * 100:.2f}%\n")

