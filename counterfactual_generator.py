import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from itertools import chain
from pprint import pprint
from statistics import mean, stdev
import os

from scipy import stats
import networkx as nx
from community import community_louvain
from CAFE import utils


from kg_info import KGENtitiesRelationsInfo
from graph_analysis import CommunityAnalyzer
from graph_analysis import *


import sys
sys.path.append('CAFE')


class Outliers:
    def __init__(self) -> None:
        pass


class EntityFilter:
    def __init__(self):
        self.kg = utils.load_kg('beauty')
        self.analyzer = CommunityAnalyzer()
        self.kg_info = KGENtitiesRelationsInfo()
        self.outlier_detector = Outliers()
        self.connection_stats, _ = self.kg_info.calculate_connection_stats()
        self.Node2Partition = self.analyzer.identify_communities()
        self.partition2Node = {}
        for key, value in self.Node2Partition.items():
            if value not in self.partition2Node:
                self.partition2Node[value] = []
            self.partition2Node[value].append(key)

    def set_community_flag(self, h_entity_type, t_entity_type, h_entity_id, zscore_threshold=0.2, zscore=True):
        if zscore:
            z_score = self.connection_stats[(h_entity_type, t_entity_type)]['z_scores'].get(h_entity_id, float('inf'))
            if z_score > zscore_threshold:
                return True
        centrality = self.outlier_checker.centrality_dict[f'{h_entity_type}_{h_entity_id}']
        return self.outlier_checker.is_outlier(centrality)

    def filter_entities(self, product_id, related_entities, h_entity_type, t_entity_type, h_entity_id, zscore_threshold=0.2, force_community_filter=False, drop_outlier_head=False):
        community_flag = self.set_community_flag(h_entity_type, t_entity_type, h_entity_id, zscore_threshold, True)

        if community_flag or force_community_filter:
            recommended_product_community = self.Node2Partition.get(('product_{}'.format(product_id), 'product'), None)
            if recommended_product_community is None:
                return set()  # Return an empty set if the product community is not found
            
            partition_et_ids = []
            for node, type in self.partition2Node.get(recommended_product_community, []):
                if type == t_entity_type:
                    entity_id = int(node.split('_')[-1])
                    partition_et_ids.append(entity_id)

            # partition_nodes = self.partition2Node.get(recommended_product_community, [])
            # partition_entities = {int(node[0].split('_')[-1]) for node in partition_nodes if node[0].startswith(t_entity_type)}
            centrality = lambda entity: self.outlier_checker.centrality_dict[f'{t_entity_type}_{entity}']
            return {entity for entity in related_entities if entity in partition_et_ids and not self.outlier_checker.is_outlier(centrality(entity), method="knee")}
        
        centrality = lambda entity: self.outlier_checker.centrality_dict[f'{t_entity_type}_{entity}']
        return {entity for entity in related_entities if not self.outlier_checker.is_outlier(entity, method="knee")}
    


    def update_entities(self, kg, related_products, recommended_product_id, entities, force_community_filter):
        for related_product_id in related_products:
            related_product_info = kg['product'].get(related_product_id, {})
            category_entities = related_product_info.get('belongs_to', [])
            filtered_categories = self.filter_entities(recommended_product_id, category_entities, 'product', 'category', related_product_id, force_community_filter=force_community_filter)
            entities['category'].update(filtered_categories)

            brand_entities = related_product_info.get('produced_by', [])
            filtered_brands = self.filter_entities(recommended_product_id, brand_entities, 'product', 'brand', related_product_id, force_community_filter=force_community_filter)
            entities['brand'].update(filtered_brands)

            word_entities = related_product_info.get('described_by', [])
            filtered_words = self.filter_entities(recommended_product_id, word_entities, 'product', 'word', related_product_id, force_community_filter=force_community_filter)
            entities['word'].update(filtered_words)

            related_related_products = related_product_info.get('also_bought', []) +\
                                        related_product_info.get('also_viewed', []) +\
                                        related_product_info.get('bought_together', [])
            filtered_related_products = self.filter_entities(recommended_product_id, related_related_products, 'product', 'related_product', related_product_id, force_community_filter=force_community_filter)
            entities['related_product'].update(filtered_related_products)

    def __call__(self, recommended_product_id, force_community_filter=False, print_report=False, outlier_method='std_dev'):
        entities = {'category': set(), 'brand': set(), 'word': set(), 'related_product': set()}
        self.outlier_checker = CentralityOutlierChecker(method = outlier_method)

        category_ids = self.kg.G['product'].get(recommended_product_id, {}).get('belongs_to', [])
        filtered_category_ids = self.filter_entities(recommended_product_id, category_ids, 'product', 'category', recommended_product_id)
        for category_id in filtered_category_ids:
            related_products = self.kg.G['category'].get(category_id, {}).get('rev_belongs_to', [])
            filtered_related_products = self.filter_entities(recommended_product_id, related_products, 'category', 'product', category_id, force_community_filter=force_community_filter)
            self.update_entities(self.kg.G, filtered_related_products, recommended_product_id, entities, force_community_filter)


        # Process brands
        brand_ids = self.kg.G['product'].get(recommended_product_id, {}).get('produced_by', [])
        filtered_brand_ids = self.filter_entities(recommended_product_id, brand_ids, 'product', 'brand', recommended_product_id)
        for brand_id in filtered_brand_ids:
            related_products = self.kg.G['brand'].get(brand_id, {}).get('rev_produced_by', [])
            filtered_related_products = self.filter_entities(recommended_product_id, related_products, 'brand', 'product', brand_id, force_community_filter=force_community_filter)
            self.update_entities(self.kg.G, filtered_related_products, recommended_product_id, entities, force_community_filter)

        # Process words
        word_ids = self.kg.G['product'].get(recommended_product_id, {}).get('described_by', [])
        filtered_word_ids = self.filter_entities(recommended_product_id, word_ids, 'product', 'word', recommended_product_id)
        for word_id in filtered_word_ids:
            related_products = self.kg.G['word'].get(word_id, {}).get('rev_described_by', [])
            filtered_related_products = self.filter_entities(recommended_product_id, related_products, 'word', 'product', word_id, force_community_filter=force_community_filter)
            self.update_entities(self.kg.G, filtered_related_products, recommended_product_id, entities, force_community_filter)

        # Process directly related products
        related_products_ids = self.kg.G['product'].get(recommended_product_id, {}).get('also_bought', []) +\
                                self.kg.G['product'].get(recommended_product_id, {}).get('also_viewed', []) +\
                                self.kg.G['product'].get(recommended_product_id, {}).get('bought_together', [])
        filtered_related_products_ids = self.filter_entities(recommended_product_id, related_products_ids, 'product', 'related_product', recommended_product_id, force_community_filter=force_community_filter)
        self.update_entities(self.kg.G, filtered_related_products_ids, recommended_product_id, entities, force_community_filter)


        entites= {k: list(v) for k, v in entities.items()}
        if print_report:
            self.report_related_entities_statistics(entites)

        tmp_directory = 'tmp'
        if not os.path.exists(tmp_directory):
            os.makedirs(tmp_directory)

        # filename = f'tmp/counter_paths_scores{product_id}.pkl'
        # # Open a file in write-binary mode
        # with open(filename, 'wb') as file:
        #     # Serialize the dictionary using pickle.dump
        #     pickle.dump(entites, file)

        return entites

    def report_related_entities_statistics(self, entities):
        """
        Generates a report on the statistics of the related entities for a given product.
        """

        print("Related Entities Statistics Report:\n")
        
        # Categories
        num_related_categories = len(entities['category'])
        num_kg_categories = len(self.kg.G['category'])
        print(f"Number of related categories: {num_related_categories}")
        print(f"Number of categories in the knowledge graph: {num_kg_categories}")
        print(f"Percentage of related categories: {num_related_categories / num_kg_categories * 100:.2f}%\n")
        
        # Brands
        num_related_brands = len(entities['brand'])
        num_kg_brands = len(self.kg.G['brand'])
        print(f"Number of related brands: {num_related_brands}")
        print(f"Number of brands in the knowledge graph: {num_kg_brands}")
        print(f"Percentage of related brands: {num_related_brands / num_kg_brands * 100:.2f}%\n")
        
        # Words
        num_related_words = len(entities['word'])
        num_kg_words = len(self.kg.G['word'])
        print(f"Number of related words: {num_related_words}")
        print(f"Number of words in the knowledge graph: {num_kg_words}")
        print(f"Percentage of related words: {num_related_words / num_kg_words * 100:.2f}%\n")
        
        # Related Products
        num_related_products = len(entities['related_product'])
        num_kg_related_products = len(self.kg.G['related_product'])
        print(f"Number of related products: {num_related_products}")
        print(f"Number of products in the knowledge graph: {num_kg_related_products}")
        print(f"Percentage of related products: {num_related_products / num_kg_related_products * 100:.2f}%\n")



# # Create an instance of EntityFilter and generate neighbor_entities
# filter = EntityFilter()
# neighbor_entities = filter(0, force_community_filter=True, print_report=True)

# # Specify the filename where you want to save the dictionary
# filename = 'neighbor_entities.pkl'

# # Open a file in write-binary mode
# with open(filename, 'wb') as file:
#     # Serialize the dictionary using pickle.dump
#     pickle.dump(neighbor_entities, file)
