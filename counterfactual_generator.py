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

class EntityFilter:
    def __init__(self):
        self.kg = utils.load_kg('beauty')
        self.analyzer = CommunityAnalyzer(self.kg.G)
        self.kg_info = KGENtitiesRelationsInfo()
        self.connection_stats, _ = self.kg_info.calculate_connection_stats()
        self.Node2Partition = self.analyzer.identify_communities()
        self.partition2Node = {}
        for key, value in self.Node2Partition.items():
            if value not in self.partition2Node:
                self.partition2Node[value] = []
            self.partition2Node[value].append(key)

    def community_filter(self, recommended_product_id, related_entities, h_entity_id, h_entity_type, t_entity_type, zscore_threshold=0.02, force_community_filter=False):
        # Get z-score of the entity
        entity_zscore = self.connection_stats[(h_entity_type, t_entity_type)]['z_scores'][h_entity_id]

        if entity_zscore <= zscore_threshold and not force_community_filter:
            return set(related_entities)
        else:
            # Perform community filtering
            recommended_product_community = self.Node2Partition[('product_{}'.format(recommended_product_id), 'product')]
            partition_nodes = self.partition2Node[recommended_product_community]
            partition_entities = {int(node[0].split('_')[-1]) for node in partition_nodes if node[1] == t_entity_type}
            related_entities_in_partition = {rel_ent for rel_ent in related_entities if rel_ent in partition_entities}

            return related_entities_in_partition

    def filter_entities(self, entity_type, related_entities, product_id, related_product_id, force_community_filter=False):
        entity_zscore = self.connection_stats[('product', entity_type)]['z_scores'][related_product_id]
        if entity_zscore > 1 or force_community_filter:
            related_entities = self.community_filter(product_id, related_entities, related_product_id, 'product', entity_type, force_community_filter=force_community_filter)
        # TODO: you can add filtering the finals based on betweenness centrality
        return related_entities

    def __call__(self, product_id, force_community_filter=False, print_report=False):
        entities = {'category': set(), 'brand': set(), 'word': set(), 'related_product': set()}

        # Process categories
        category_ids = self.kg.G['product'].get(product_id, {}).get('belongs_to', [])
        for category_id in category_ids:
            related_products = self.kg.G['category'].get(category_id, {}).get('rev_belongs_to', [])
            related_products = self.community_filter(product_id, related_products, category_id, 'category', 'product', force_community_filter=force_community_filter)

            for related_product_id in related_products:
                related_product_info = self.kg.G['product'].get(related_product_id, {})
                entities['category'].update(self.filter_entities('category', related_product_info.get('belongs_to', []), product_id, related_product_id, force_community_filter=force_community_filter))
                entities['brand'].update(self.filter_entities('brand', related_product_info.get('produced_by', []), product_id, related_product_id, force_community_filter=force_community_filter))
                entities['word'].update(self.filter_entities('word', related_product_info.get('described_by', []), product_id, related_product_id, force_community_filter=force_community_filter))
                related_related_products = related_product_info.get('also_bought', []) +\
                                           related_product_info.get('also_viewed', []) +\
                                           related_product_info.get('bought_together', [])
                entities['related_product'].update(self.filter_entities('related_product', related_related_products, product_id, related_product_id, force_community_filter=force_community_filter))

        # Process brands
        brand_ids = self.kg.G['product'].get(product_id, {}).get('produced_by', [])
        for brand_id in brand_ids:
            related_products = self.kg.G['brand'].get(brand_id, {}).get('rev_produced_by', [])
            related_products = self.community_filter(product_id, related_products, brand_id, 'brand', 'product', force_community_filter=force_community_filter)

            for related_product_id in related_products:
                related_product_info = self.kg.G['product'].get(related_product_id, {})
                entities['category'].update(self.filter_entities('category', related_product_info.get('belongs_to', []), product_id, related_product_id, force_community_filter=force_community_filter))
                entities['brand'].update(self.filter_entities('brand', related_product_info.get('produced_by', []), product_id, related_product_id, force_community_filter=force_community_filter))
                entities['word'].update(self.filter_entities('word', related_product_info.get('described_by', []), product_id, related_product_id, force_community_filter=force_community_filter))
                related_related_products = related_product_info.get('also_bought', []) +\
                                           related_product_info.get('also_viewed', []) +\
                                           related_product_info.get('bought_together', [])
                entities['related_product'].update(self.filter_entities('related_product', related_related_products, product_id, related_product_id, force_community_filter=force_community_filter))

        # Process words
        word_ids = self.kg.G['product'].get(product_id, {}).get('described_by', [])
        for word_id in word_ids:
            related_products = self.kg.G['word'].get(word_id, {}).get('rev_described_by', [])
            related_products = self.community_filter(product_id, related_products, word_id, 'word', 'product', force_community_filter=force_community_filter)

            for related_product_id in related_products:
                related_product_info = self.kg.G['product'].get(related_product_id, {})
                entities['category'].update(self.filter_entities('category', related_product_info.get('belongs_to', []), product_id, related_product_id, force_community_filter=force_community_filter))
                entities['brand'].update(self.filter_entities('brand', related_product_info.get('produced_by', []), product_id, related_product_id, force_community_filter=force_community_filter))
                entities['word'].update(self.filter_entities('word', related_product_info.get('described_by', []), product_id, related_product_id, force_community_filter=force_community_filter))
                related_related_products = related_product_info.get('also_bought', []) +\
                                           related_product_info.get('also_viewed', []) +\
                                           related_product_info.get('bought_together', [])
                entities['related_product'].update(self.filter_entities('related_product', related_related_products, product_id, related_product_id, force_community_filter=force_community_filter))

        # Process related products
        related_products_1 = self.kg.G['product'].get(product_id, {}).get('also_bought', []) 
        related_products_2 = self.kg.G['product'].get(product_id, {}).get('also_viewed', []) 
        related_products_3 = self.kg.G['product'].get(product_id, {}).get('bought_together', []) 
        related_products = related_products_1 + related_products_2 + related_products_3

        related_products = self.community_filter(product_id, related_products, product_id, 'related_product', 'product', force_community_filter=force_community_filter)

        for related_product_id in related_products:
            related_product_info = self.kg.G['product'].get(related_product_id, {})
            entities['category'].update(self.filter_entities('category', related_product_info.get('belongs_to', []), product_id, related_product_id, force_community_filter=force_community_filter))
            entities['brand'].update(self.filter_entities('brand', related_product_info.get('produced_by', []), product_id, related_product_id, force_community_filter=force_community_filter))
            entities['word'].update(self.filter_entities('word', related_product_info.get('described_by', []), product_id, related_product_id, force_community_filter=force_community_filter))
            related_related_products = related_product_info.get('also_bought', []) +\
                                       related_product_info.get('also_viewed', []) +\
                                       related_product_info.get('bought_together', [])
            entities['related_product'].update(self.filter_entities('related_product', related_related_products, product_id, related_product_id, force_community_filter=force_community_filter))

        entites= {k: list(v) for k, v in entities.items()}
        if print_report:
            self.report_related_entities_statistics(entites)

        tmp_directory = 'tmp'
        if not os.path.exists(tmp_directory):
            os.makedirs(tmp_directory)

        filename = f'tmp/counter_paths_scores{product_id}.pkl'
        # Open a file in write-binary mode
        with open(filename, 'wb') as file:
            # Serialize the dictionary using pickle.dump
            pickle.dump(entites, file)

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
