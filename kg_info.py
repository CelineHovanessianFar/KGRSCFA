import pandas as pd
from collections import defaultdict
from statistics import mean, stdev
import numpy as np
from scipy import stats as stat_module
from pprint import pprint
from CAFE import utils
import pickle



class KGENtitiesRelationsInfo:
    def __init__(self, args=None):
        if args is None:
            dataset = 'beauty'  
        else:
            dataset = args.dataset

        self.kg = utils.load_kg(dataset)  
        # TODO: does this really exist? could be defined in this class instead of kg
        self.ent2related_ent = self.kg.entity_related_type_info  # Map each entity type to connected entity types.
        self.old2new_ids = self._get_old_to_new_ids()  # Map old entity IDs (before KG shrinkage) to new ones.
        self.new2old_ids = self._get_new_to_old_ids()  # Map new entity IDs (after KG shrinkage) back to the old ones.
        self.id2entity_name = self.load_ids_dictionary(dataset)  # Map entity IDs to product names.
        self.product2attribute_rels = self._get_entity_to_attribute_rels()  # Map each entity connected to the 'product' entity type to the relationships that link them.
        
        # self.mp2entities = self._mp2ents_generate()  # Map each metapath (mp) to the types of entities it includes.
        # self.relations2mp = self._relation2mp_generate()  # Map tuples of relations to their corresponding metapath IDs.
        # self.ent2relations = self._entity_relations_info()  # Map each entity type to all the relations it can have.



    def _mp2ents_generate(self):
        mp2ents = dict()
        for i, path in enumerate(self.kg.metapaths):
            mp2ents[i] = [pair[1] for pair in path]
        return mp2ents

    def _mp2relations_generate(self):
        mp2relations = dict()
        for i, path in enumerate(self.kg.metapaths):
            mp2relations[i] = [pair[0] for pair in path]
        return mp2relations

    def _relation2mp_generate(self):
        mp2relation = self._mp2relations_generate()
        return {tuple(products[1:]): i for i, products in mp2relation.items()}

    def _entity_relations_info(self):
        entity2relation = {}
        for relation, related_pair in self.kg.relation_info.items():
            entity = related_pair[0]
            if entity not in entity2relation:
                entity2relation[entity] = []
            entity2relation[entity].append(relation)
        return entity2relation

    def calculate_connection_stats(self):
        """
        Calculate statistical metrics for connections between entities in the knowledge graph.

        This method analyzes the connections between entities of different types in the KG, 
        calculates the count of related entities for each entity, and computes statistical 
        measures like mean, standard deviation, z-scores, and outlier detection.

        Returns:
            tuple: A tuple containing:
                - connection_stats (dict): A nested dictionary where each key is a tuple 
                (entity_type, related_entity_type) and values are statistics:
                    - 'counts': Mapping of entity IDs to their connection counts.
                    - 'mean': Mean number of connections for the entity type with related_entity_type.
                    - 'std_dev': Standard deviation of connection counts.
                    - 'lower_bound': Lower bound for non-outlier connections.
                    - 'upper_bound': Upper bound for non-outlier connections.
                    - 'z_scores': Z-scores for each entity's connection count.
                    - 'outliers': List of entity IDs identified as outliers.
                - connection_stats_df (pd.DataFrame): A DataFrame summarizing the connection
                statistics with columns:
                    - 'h_entity': Source entity type.
                    - 't_entity_type': Target entity type.
                    - 'h_entity_id': Source entity ID.
                    - 'count': Connection count for the entity.
                    - 'mean': Mean connection count for the entity type.
                    - 'std_dev': Standard deviation of connection counts.
                    - 'upper_bound': Upper bound for outliers.
                    - 'z_score': Z-score of the connection count.
                    - 'is_outlier': Boolean indicating if the entity is an outlier.

        Process:
            1. Counts the number of connections each entity has to related entities, grouped by 
            source and target entity types.
            2. Aggregates the connection counts and calculates summary statistics:
            - Mean and standard deviation.
            - Z-scores for identifying outliers.
            - Boundaries for non-outlier detection based on z-scores.
            3. Flags entities with z-scores outside the range [-3, 3] as outliers.
            4. Compiles the statistics into both a nested dictionary and a Pandas DataFrame for analysis.
        """        
        temp = defaultdict(lambda: defaultdict(list))
        connection_stats = {}

        for entity in self.kg.G:
            for eid, ent_info in self.kg.G[entity].items():
                for relation, related_ids in ent_info.items():
                    _, et_type = self.kg.relation_info[relation]
                    temp[(entity, et_type)][eid].append(len(related_ids))
                    # TODO: could be:
                    # temp[(entity, et_type)][eid] = len(related_ids)
                    # how many related et_type each eid of entity type is connected to


        # TODO: this looks redundant
        # Aggregate counts for each (entity, et_type)
        aggregated_counts = defaultdict(lambda: defaultdict(int))

        for h_entity_type in self.kg.G:
            for eid in self.kg.G[h_entity_type]:
                for t_entitiy_type in self.ent2related_ent[h_entity_type]:
                    aggregated_counts[(h_entity_type, t_entitiy_type)][eid] = 0

        for (entity, et_type), eid_dict in temp.items():
            for eid, lengths in eid_dict.items():
                # TODO: could have more than one length?
                aggregated_counts[(entity, et_type)][eid] = sum(lengths)

        # Compute mean, standard deviation, and bounds for outliers for each (entity, et_type)
        for (entity, et_type), counts_dict in aggregated_counts.items():
            counts = list(counts_dict.values())
            if counts:
                mean_length = mean(counts)
                std_dev_length = stdev(counts) if len(counts) > 1 else 0.0

                # Calculate z-scores
                z_scores = stat_module.zscore(counts) if len(counts) > 1 else [0] * len(counts)

                # Define bounds for outliers
                lower_bound = mean_length - 3 * std_dev_length
                upper_bound = mean_length + 3 * std_dev_length

                connection_stats[(entity, et_type)] = {
                    'counts': counts_dict,
                    'mean': mean_length,
                    'std_dev': std_dev_length,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'z_scores': {},
                    'outliers': []
                }

                for eid, count in counts_dict.items():
                    # Find the z-score corresponding to this count
                    idx = counts.index(count)
                    z_score = z_scores[idx]
                    connection_stats[(entity, et_type)]['z_scores'][eid] = z_score

                    # Check if the entity ID is an outlier
                    # TODO: 3 is hard-coded
                    if z_score < -3 or z_score > 3:
                        connection_stats[(entity, et_type)]['outliers'].append(eid)

        # Create records for DataFrame
        records = []
        for (entity, et_type), stats in connection_stats.items():
            for eid, count in stats['counts'].items():
                record = {
                    'h_entity': entity,
                    't_entity_type': et_type,
                    'h_entity_id': eid,
                    'count': count,
                    'mean': stats['mean'],
                    'std_dev': stats['std_dev'],
                    'upper_bound': stats['upper_bound'],
                    'z_score': stats['z_scores'][eid],
                    'is_outlier': eid in stats['outliers']
                }
                records.append(record)

        connection_stats_df = pd.DataFrame(records)
        return connection_stats, connection_stats_df
    
    def _get_new_to_old_ids(self):
        with open('CAFE/tmp/new_to_old.pickle', 'rb') as file:
            new_to_old = pickle.load(file)
        return new_to_old

    def _get_old_to_new_ids(self):
        with open('CAFE/tmp/old_to_new.pickle', 'rb') as file:
            old_to_new = pickle.load(file)
        return old_to_new

    def load_ids_dictionary(self, dataset):
        filepath = utils.DATA_DIR[dataset] + '/id2entity.pickle'
        with open(filepath, 'rb') as file:
            id2entity = pickle.load(file)
        return id2entity

    def _get_entity_to_attribute_rels(self):
        attributes_mapping = {
            'word': 'described_by',
            'brand': 'produced_by',
            'category': 'belongs_to',
            'related_product': ['also_viewed', 'also_bought', 'bought_together']
        }
        return attributes_mapping        


