import pandas as pd
from collections import defaultdict
from statistics import mean, stdev
import numpy as np
from scipy import stats as stat_module
from pprint import pprint
from CAFE import utils


class KGENtitiesRelationsInfo:
    def __init__(self):
        self.kg = utils.load_kg('beauty')
        self.relation_info = self.kg.relation_info
        self.metapaths = self.kg.metapaths
        self.mp2ents = self._mp2ents_generate()
        self.rel2mp = self._relation2mp_generate()
        self.ent2related_ent = self.kg.entity_related_type_info
        self.ent2rel = self._entity_relations_info()

    def _mp2ents_generate(self):
        mp2ents = dict()
        for i, path in enumerate(self.metapaths):
            mp2ents[i] = [pair[1] for pair in path]
        return mp2ents

    def _mp2relations_generate(self):
        mp2relations = dict()
        for i, path in enumerate(self.metapaths):
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
        temp = defaultdict(lambda: defaultdict(list))
        connection_stats = {}

        for entity in self.kg.G:
            for eid, ent_info in self.kg.G[entity].items():
                for relation, related_ids in ent_info.items():
                    _, et_type = self.relation_info[relation]
                    temp[(entity, et_type)][eid].append(len(related_ids))

        # Aggregate counts for each (entity, et_type)
        aggregated_counts = defaultdict(lambda: defaultdict(int))

        for h_entity_type in self.kg.G:
            for eid in self.kg.G[h_entity_type]:
                for t_entitiy_type in self.ent2related_ent[h_entity_type]:
                    aggregated_counts[(h_entity_type, t_entitiy_type)][eid] = 0

        for (entity, et_type), eid_dict in temp.items():
            for eid, lengths in eid_dict.items():
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