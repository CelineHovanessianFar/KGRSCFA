import os
import pickle
import sys

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

sys.path.append('CAFE')

from CAFE import utils
from CAFE.execute_neural_symbol import create_symbolic_model, MetaProgramExecutor, create_heuristic_program, infer_paths
from counterfactual_generator import EntityFilter
from kg_info import KGENtitiesRelationsInfo

from data_registry import DataRegistry


class CFAnalyzer:
    def __init__(self, recommendations, recommendation_to_analyze, args) -> None:
        # TODO: is self.args a good approach?
        self.args = args
        self.recommendation_to_analyze = recommendation_to_analyze
        self.recommendations = recommendations
        self.cf_scenario_generator = CFScenarioGenerator()
        self.cf_curator = CFCurator()
        self.analysis_reporter = AnalysisReport()

    def analyze(self):
        cf_paths_scores = self.cf_scenario_generator.generate_counterfactual_scenarios(self.args)
        eligible_cf_entities = self.cf_curator.curate_eligible_cf_scenarios(cf_paths_scores)
        self.analysis_reporter.report_qualified_cf_scenarios(eligible_cf_entities)
    
    def generate_counterfactual_scenarios(self):
        return self.cf_scenario_generator.generate_counterfactual_scenarios(self.args)

    def curate_eligible_cf_scenarios(self):
        cf_paths_scores = self.generate_counterfactual_scenarios()
        return self.cf_curator.curate_eligible_cf_scenarios(cf_paths_scores)

    
class CFScenarioGenerator:
    def __init__(self):
        pass

    def generate_counterfactual_scenarios(self, args):
        recommended_product_id = DataRegistry.recommendation_to_analyze[0][-1]
        user_id = DataRegistry.recommendation_to_analyze[0][0]

        cf_paths_scores_filename = self.get_score_filename(user_id, recommended_product_id)

        if os.path.exists(cf_paths_scores_filename):
            return self.load_scores(cf_paths_scores_filename)
        
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            user_purchases = self.get_user_purchases(user_id)
            neighboring_entities = self.extract_neighbors(recommended_product_id)
            model = self.initialize_model(args)
            cf_paths_scores = self.generate_and_score_cf_paths(user_purchases, neighboring_entities, model)
            # Save scores for future use
            self.save_scores(user_id, recommended_product_id, cf_paths_scores)

        return cf_paths_scores

    def get_user_purchases(self, user_id):
        return DataRegistry.kg.G['user'].get(user_id, {}).get('purchase', [])

    def extract_neighbors(self, recommended_product_id):
        filename = f'tmp/neighbor_entities_{recommended_product_id}.pkl'
        try:
            with open(filename, 'rb') as file:
                neighbors=pickle.load(file)
            print('loaded neighbors file.')
            return neighbors
        except FileNotFoundError:
            print('Finding Neighbors!')
            filter = EntityFilter()
            neighboring_entities = filter(recommended_product_id, force_community_filter=True, print_report=True)
            with open(filename, 'wb') as file:
                pickle.dump(neighboring_entities, file)
            return neighboring_entities

    def generate_and_score_cf_paths(self, user_purchases, neighboring_entities, model):
        cf_paths_scores = {}
        product_related_ents = DataRegistry.kg_info.ent2related_ent.get('product', set()) - {'user'}

        for purchased_product in tqdm(user_purchases, desc="Processing Purchased Products"):
            for entity in product_related_ents:
                related_ent_ids = neighboring_entities[entity]
                cf_metapaths = self.entity_to_metapath(entity)
                self.score_cf_metapaths(purchased_product, entity, cf_metapaths, related_ent_ids, model, cf_paths_scores)

        return cf_paths_scores

    def score_cf_metapaths(self, purchased_product, entity, cf_metapaths, related_ent_ids, model, cf_paths_scores):
        print(f"begin: {entity}")
        print(f"begin: {cf_metapaths}")
        for mp_id, counter_mp in enumerate(cf_metapaths):
            mp_temp_dict = {'mp': counter_mp}

            # Combine score_metapaths functionality here
            for rel_ent_id in related_ent_ids:
                path_ent_ids = [
                    DataRegistry.recommendation_to_analyze[0][0],  # user_id
                    purchased_product,
                    rel_ent_id,
                    DataRegistry.recommendation_to_analyze[0][-1]  # product_id
                ]
                layer_logprobs = self.infer_path_score(model, counter_mp, path_ent_ids)
                mp_temp_dict[rel_ent_id] = {'log_probs': [item.detach().cpu().numpy()[0] for item in layer_logprobs]}

            # Organize counter_scores
            if entity not in cf_paths_scores:
                cf_paths_scores[entity] = {}
            if mp_id not in cf_paths_scores[entity]:
                cf_paths_scores[entity][mp_id] = []
            cf_paths_scores[entity][mp_id].append(mp_temp_dict)

    def infer_path_score(self, model, metapath, path_ent_ids, excluded_pids=None):
        # Ensure the number of modules matches the number of path entity IDs minus one.
        modules = model._get_modules(metapath)
        assert len(modules) == len(path_ent_ids) - 1, "Mismatch in the number of modules and path entity IDs."

        # Prepare the input tensor and perform a forward pass through the model.
        uid = path_ent_ids[0]
        uid_tensor = torch.LongTensor([uid]).to(self.device)
        outputs = model._forward(modules, uid_tensor)  # list of tensor of shape [1, d]

        # Compute log probabilities for each module output.
        layer_logprobs = []
        for i, module in enumerate(modules):
            et_vecs = model.embedding(module.et_name)
            scores = torch.matmul(outputs[i], et_vecs.t())  # Shape: [1, vocab_size]
            logprobs = F.log_softmax(scores[0], dim=0)  # Shape: [vocab_size]
            valid_et_ids = torch.LongTensor([path_ent_ids[i+1]]).to(self.device)
            et_logprobs = logprobs.index_select(0, valid_et_ids)
            layer_logprobs.append(et_logprobs)

        return layer_logprobs

    def initialize_model(self, args):
        return create_symbolic_model(args, DataRegistry.kg, train=False)

    def entity_to_metapath(self, entity):
        # Define metapath templates for different entities
        entity_to_mp = {
            'brand': [[(None, 'user'), ('purchase', 'product'), ('produced_by', 'brand'), ('rev_produced_by', 'product')]],
            'category': [[(None, 'user'), ('purchase', 'product'), ('belongs_to', 'category'), ('rev_belongs_to', 'product')]],
            'word': [[(None, 'user'), ('purchase', 'product'), ('described_by', 'word'), ('rev_described_by', 'product')]],
            'related_product': [
                # Multiple metapaths for related_product
                [(None, 'user'), ('purchase', 'product'), (relation, 'related_product'), (reverse_relation, 'product')]
                for relation, reverse_relation in [
                    ('also_bought', 'rev_also_bought'),
                    ('also_bought', 'rev_also_viewed'),
                    ('also_bought', 'rev_bought_together'),
                    ('also_viewed', 'rev_also_bought'),
                    ('also_viewed', 'rev_also_viewed'),
                    ('also_viewed', 'rev_bought_together'),
                    ('bought_together', 'rev_also_bought'),
                    ('bought_together', 'rev_also_viewed'),
                    ('bought_together', 'rev_bought_together')
                ]
            ]
        }
        return entity_to_mp[entity]
        # return entity_to_mp.get(entity, ValueError(f"Unknown entity type: {entity}"))

    def load_scores(self, scores_path):
        with open(scores_path, 'rb') as file:
            return pickle.load(file)

    def save_scores(self, user_id, recommended_product_id, scores):
        filename = self.get_score_filename(user_id, recommended_product_id)
        with open(filename, 'wb') as file:
            pickle.dump(scores, file)

    def get_score_filename(self, user_id, recommended_product_id):
        return f"tmp/raw_counter_scores_{user_id}_{recommended_product_id}.pkl"


class CFCurator:
    def __init__(self):
        pass

    def curate_eligible_cf_scenarios(self, cf_paths_scores, kth=10):
        min_score = self.calculate_min_score()
        cf_entity_eid_scores = self.get_relevant_data(cf_paths_scores)
        eligible_cf_entities = self.filter_qualified_cf_scenarios(cf_entity_eid_scores, min_score)
        return eligible_cf_entities

    # fixed
    def calculate_min_score(self):
        recommended_paths_scores = [item[1] for item in DataRegistry.recommendations]
        return sorted([np.mean(item) for item in recommended_paths_scores], reverse=True)[:DataRegistry.k][-1]

    def get_relevant_data(self, cf_paths_scores):
        cf_entity_eid_scores = dict()
        for entity, mpid_dict in cf_paths_scores.items():
            for mpid, purchased_products_dicts_list in mpid_dict.items():
                for purchased_product_cf_scores_dict in purchased_products_dicts_list:
                    # Remove 'mp' key if it exists in eid_score
                    purchased_product_cf_scores_dict.pop('mp', None)
                    
                    # Ensure the entity key exists and is a list
                    if entity not in cf_entity_eid_scores:
                        cf_entity_eid_scores[entity] = []  # Initialize as an empty list
                    
                    # Append the eid_score
                    cf_entity_eid_scores[entity].append(purchased_product_cf_scores_dict)
        
        return cf_entity_eid_scores


    def filter_qualified_cf_scenarios(self, cf_entity_eid_scores, min_score):
        # Calculate the mean of log probabilities for each entity and store it in the 'mean' field
        for entity in cf_entity_eid_scores:
            for purches_product_scores_dict in cf_entity_eid_scores[entity]:
                for eid in purches_product_scores_dict: 
                    purches_product_scores_dict[eid]['mean'] = np.mean(purches_product_scores_dict[eid]['log_probs'])

        # Sort the entities by their 'mean' score in descending order
        # sorted_cf_entity_eid_scores = dict(
        #     sorted(
        #         cf_entity_eid_scores.items(),
        #         key=lambda entity: np.mean([purchased_product_scores_dict[eid]['mean'] for purchased_product_scores_dict in entity[1] for eid in purchased_product_scores_dict]),
        #         reverse=True
        #     )
        # )

        eligible_cf_entities = {}

        for entity in cf_entity_eid_scores:
            entity_eligible_cf_entities = []  # Initialize the list for each entity

            for purchased_product_scores_dict in cf_entity_eid_scores[entity]:
                # Iterate over each item in the series dictionary
                for eid, scores in purchased_product_scores_dict.items():
                    # Check if any 'mean' score is greater than min_score
                    if scores['mean'] > min_score:
                        entity_eligible_cf_entities.append((eid, scores))

            # Extend the eligible entities dictionary
            eligible_cf_entities.setdefault(entity, []).extend(entity_eligible_cf_entities)
        return eligible_cf_entities


class AnalysisReport:
    def __init__(self) -> None:
        # Use DataRegistry attributes
        self.recommended_product_id = DataRegistry.recommendation_to_analyze[0][-1]
        self.purchased_product_id = DataRegistry.recommendation_to_analyze[0][-1]
        self.caused_relation = DataRegistry.recommendation_to_analyze[-1][-2]
        self.caused_entity_type = DataRegistry.kg.relation_info[self.caused_relation][1]
        self.caused_entity_id = DataRegistry.recommendation_to_analyze[0][-2]
        self.caused_entity_old_id = DataRegistry.kg_info.new2old_ids[self.caused_entity_type][self.caused_entity_id]

    def report_qualified_cf_scenarios(self, eligible_cf_entities):
        self.eligible_cf_entities = eligible_cf_entities

        # Fetch attributes for purchased and recommended products
        purchased_product_attributes = self.get_entity_attributes(self.purchased_product_id)
        recommended_product_attributes = self.get_entity_attributes(self.recommended_product_id)

        # Display attributes
        self.print_entity_attributes(purchased_product_attributes, "purchased product")
        self.print_entity_attributes(recommended_product_attributes, "recommended product")

        # Plausible Counterfactuals (now using the same print method)

        # eligible_attributes = {
        #     entity: [(id[0], id[1]) for id in self.eligible_cf_entities.get(entity, [])]
        #     for entity in ['word', 'brand', 'category', 'related_product']
        # }
        self.print_entity_attributes(self.eligible_cf_entities, "eligible counterfactuals", include_scores=False, coutnerfactual_mode=True)

    def get_entity_attributes(self, product_id):
        attributes = {}
        kg = DataRegistry.kg

        
        for attr, rels in DataRegistry.kg_info.product2attribute_rels.items():
            if isinstance(rels, list):  # Combine multiple relationships
                combined_ids = []
                for rel in rels:
                    combined_ids.extend(kg.G['product'][product_id].get(rel, []))
                attributes[attr] = combined_ids
            else:
                attributes[attr] = kg.G['product'][product_id].get(rels, [])

        return attributes

    def print_entity_attributes(self, attributes, title, include_scores=False, num_columns=5, coutnerfactual_mode=False, num_stars=170):
        def print_stars(num_stars=num_stars):
            print("*" * num_stars)
        
        print_stars()
        print(f"Attributes of the {title}:")
        print_stars()

        # Use DataRegistry's kg_info and id2entity attributes
        kg_info = DataRegistry.kg_info
        # id2entity = DataRegistry.kg_info.id2entity
        id2entity = DataRegistry.kg_info.id2entity_name

        for entity, ids in attributes.items():
            if ids:
                # Safeguarding against unhashable types
                filtered_ids = [id[0] if isinstance(id, tuple) and isinstance(id[0], (int, str)) else id for id in ids]
                old_ids = [kg_info.new2old_ids[entity][id] for id in filtered_ids]
                entities = {id2entity[entity][old_id] for old_id in old_ids}
                
                # TODO:this part is fishy, debug with include_scores
                if include_scores:
                    entities = [f"{id2entity[entity][old_id]} (Score: {score})"
                                for old_id, score in zip(old_ids, [id[1] for id in ids if isinstance(id, tuple) and len(id) == 2])]
                else:
                    entities = {id2entity[entity][old_id] for old_id in old_ids}

                # Checking the mode to decide the header format
                if not coutnerfactual_mode:
                    # Print header for each entity type
                    if entity == 'word':
                        print("WORDS the product is described by: \n")
                    elif entity == 'brand':
                        print('BRAND of the Product is: \n')
                    elif entity == 'category':
                        print("CATEGORIES the product belongs to: \n")
                    elif entity == 'related_product':
                        print("RELATED PRODUCTS that have been bought together, also bought, or also viewed with this product: \n")
                else:
                    # Print hypothetical headers for each entity type
                    if entity == 'word':
                        print("If the product had these WORDS, it would still be recommended: \n")
                    elif entity == 'brand':
                        print('If the product had this BRAND, it would still be recommended: \n')
                    elif entity == 'category':
                        print("If the product belonged to these CATEGORIES, it would still be recommended: \n")
                    elif entity == 'related_product':
                        print("If these were the RELATED PRODUCTS, the product would still be recommended: \n")

                # Calculate the format width based on the longest entity and column number
                longest_entity = max((len(entity) for entity in entities), default=0)
                column_width = 28

                # Print entities in specified number of columns
                for index, entity_name in enumerate(entities, start=1):
                    print(f"{entity_name:{column_width}}", end='')
                    if index % num_columns == 0:
                        print()  # Newline after the specified number of columns

                # If the last line of entities doesn't fill the full row, print a newline
                if len(entities) % num_columns != 0:
                    print()  # Ensure ending on a new line after the list
            print("\n")
