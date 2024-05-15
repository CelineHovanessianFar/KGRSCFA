from __future__ import absolute_import, division, print_function

import random
import numpy as np
import gzip
import KG_shrinkage_utils
from utils import DATA_DIR
import utils
import logging
from tqdm import tqdm
from shrink_postprocess import ShrinkPosprocess

# ENTITY TYPES
USER = 'user'
WORD = 'word'
PRODUCT = 'product'
BRAND = 'brand'
CATEGORY = 'category'
RPRODUCT = 'related_product'

# RELATIONS
PURCHASE = 'purchase'
MENTION = 'mentions'
DESCRIBED_BY = 'described_by'
PRODUCED_BY = 'produced_by'
BELONG_TO = 'belongs_to'
ALSO_BUY = 'also_bought'
ALSO_VIEW = 'also_viewed'
BUY_TOGETHER = 'bought_together'

# REVERSED RELATIONS
REV_PREFIX = 'rev_'
REV_PURCHASE = REV_PREFIX + PURCHASE
REV_MENTION = REV_PREFIX + MENTION
REV_DESCRIBED_BY = REV_PREFIX + DESCRIBED_BY
REV_PRODUCED_BY = REV_PREFIX + PRODUCED_BY
REV_BELONG_TO = REV_PREFIX + BELONG_TO
REV_ALSO_BUY = REV_PREFIX + ALSO_BUY
REV_ALSO_VIEW = REV_PREFIX + ALSO_VIEW
REV_BUY_TOGETHER = REV_PREFIX + BUY_TOGETHER

SELF_LOOP = 'self_loop'



class MyKnowledgeGraphShrinkable:
    def __init__(self, dataset, sample_numb=None):
        if sample_numb:
            self.entity_sample_limits = KG_shrinkage_utils.sample_users_and_products(sample_numb)
        self.G = dict()
        # Assume each relation correspnonds to a unique pair of (head_entity_type, tail_entity_type)!
        # E.g., purchase -> (user, item)
        # Otherwise, the KG cannot be built in this case.
        self.relation_info = dict()
        self.metapaths = list()
        self._init(dataset)
        if self.entity_sample_limits:
            self.pruner = self.PruneGraph(self)
            self.G = self.pruner.prune_graph()
            self.shrink_labels(self.G['user'].keys())
            self.posprocessor = ShrinkPosprocess(self)
            self.posprocessor.update_ids_embeddings()
            assert hasattr(self, 'user_old_to_new_id')
            # self.user_old_to_new_id, self.product_old_to_new_id = self.remap_entity_ids(self.entity_related_type_info)
            self.update_labels_ids()
            
        
    def _init(self, dataset):
        # Load entities
        entity_file = DATA_DIR[dataset] + "/kg_entities.txt.gz"
        id2entity = {}
        num_entities = 0
        with gzip.open(entity_file, "rb") as f:
            for line in f:
                # Format: [entity_global_id]\t[entity_type]_[entity_local_id]\t[entity_value]
                cells = line.decode().strip().split("\t")
                global_id = int(cells[0])
                entity_eid = cells[1].rsplit("_", maxsplit=1)
                entity, eid = entity_eid[0], int(entity_eid[1])
                if self.entity_sample_limits:
                    if entity in self.entity_sample_limits:
                        if eid not in self.entity_sample_limits[entity]:
                            continue
                if entity not in self.G:
                    self.G[entity] = {}
                self.G[entity][eid] = {}
                id2entity[global_id] = (entity, eid)
                num_entities += 1
        print(f'>>> {num_entities} entities are loaded.')

        # Load relations
        relation_file = DATA_DIR[dataset] + "/kg_relations.txt.gz"
        id2rel = {}
        with gzip.open(relation_file, "rb") as f:
            for line in f:
                # Format: [relation_global_id]\t[relation_name]
                cells = line.decode().strip().split("\t")
                rid = int(cells[0])
                rel = cells[1]
                id2rel[rid] = rel
        print(f'>>> {len(id2rel)} relations are loaded.')

        # Load triples
        triple_file = DATA_DIR[dataset] + "/kg_triples.txt.gz"
        num_triples = 0
        with gzip.open(triple_file, "rb") as f:
            for line in f:
                # Format: [head_entity_global_id]\t[relation_global_id]\t[tail_entity_global_id]
                cells = line.decode().strip().split("\t")
                if self.entity_sample_limits:
                    if int(cells[0]) not in id2entity:
                        continue
                head_ent, hid = id2entity[int(cells[0])]
                rel = id2rel[int(cells[1])]
                if self.entity_sample_limits:
                    if int(cells[2]) not in id2entity:
                        continue
                tail_ent, tid = id2entity[int(cells[2])]

                # Validate if the assumption is correct!
                if rel not in self.relation_info:
                    self.relation_info[rel] = (head_ent, tail_ent)
                else:
                    assert self.relation_info[rel] == (head_ent, tail_ent)

                # Add edge.
                if self.entity_sample_limits:
                    if tail_ent in self.entity_sample_limits:
                        if tid not in self.entity_sample_limits[tail_ent]:
                            continue    
                if rel not in self.G[head_ent][hid]:
                    self.G[head_ent][hid][rel] = []
                self.G[head_ent][hid][rel].append(tid)
                num_triples += 1
        print(f'>>> {num_triples} triples are loaded (including reverse edges).')

        # Load rules
        rule_file = DATA_DIR[dataset] + "/kg_rules.txt.gz"
        with gzip.open(rule_file) as f:
            for line in f:
                cells = line.decode().strip().split("\t")
                mp = []
                for rid in cells:
                    rel = id2rel[int(rid)]
                    head_ent, tail_ent = self.relation_info[rel]
                    if not mp:
                        mp.append((None, head_ent))
                    else:
                        assert mp[-1][1] == head_ent
                    mp.append((rel, tail_ent))
                self.metapaths.append(mp)
        print(f'>>> {len(self.metapaths)} rules are loaded.')
        # print(self.metapaths)


    class PruneGraph:
        def __init__(self, kg_obj):
            self.kg_obj = kg_obj
            self.G = self.kg_obj.G.copy()
            kg_obj.relation_info = kg_obj.relation_info
            kg_obj.entity_related_type_info = self.entity_related_type_info_generator()
            kg_obj.reversed_relation_info = self.reversed_relation_info_generator()

        def entity_related_type_info_generator(self):
            logging.info("Generating entity related type information...")
            relation_info = self.kg_obj.relation_info
            entity_related_type_info = {entity: set() for entity in self.kg_obj.G}
            for relation in relation_info:
                h_entity, t_entity = relation_info[relation]
                entity_related_type_info[h_entity].update([t_entity])
            logging.info("Entity related type information generated.")
            return entity_related_type_info

        def reversed_relation_info_generator(self):
            logging.info("Generating reversed relation information...")
            reversed_relation_info = {}
            for key, value in self.kg_obj.relation_info.items():
                if value not in reversed_relation_info:
                    reversed_relation_info[value] = [key]
                else:
                    if not isinstance(reversed_relation_info[value], list):
                        reversed_relation_info[value] = [reversed_relation_info[value]]
                    reversed_relation_info[value].append(key)
            logging.info("Reversed relation information generated.")
            return reversed_relation_info

        def identify_invalid_ents(self):
            logging.info("Identifying and removing invalid entities...")
            removed_entities = {entity: [] for entity in self.G}
            for entity in tqdm(self.G, desc="Processing Entities"):
                related_entities = self.kg_obj.entity_related_type_info[entity]
                for related_ent in related_entities:
                    valid_rel_ent_ids = self.G[related_ent].keys()
                    for relation in self.kg_obj.reversed_relation_info[(entity, related_ent)]:
                        for eid in self.G[entity].keys():
                            if relation in self.G[entity][eid]:
                                deleted_entities = [rel_eid for rel_eid in self.G[entity][eid][relation] if rel_eid not in valid_rel_ent_ids]
                                removed_entities[related_ent].extend(deleted_entities)
                                self.G[entity][eid][relation] = [rel_eid for rel_eid in self.G[entity][eid][relation] if rel_eid in valid_rel_ent_ids]
            logging.info("Invalid entities removal completed.")
            return removed_entities

        def prune_graph(self):
            logging.info("Starting graph pruning...")
            valid_ids = {entity: set() for entity in self.G}
            for entity in tqdm(['user', 'product'], desc="Pruning Users and Products"):
                related_entities = self.kg_obj.entity_related_type_info[entity]
                for related in related_entities:
                    relations = self.kg_obj.reversed_relation_info[(entity, related)]
                    for relation in relations:
                        for eid in self.G[entity]:
                            if relation in self.G[entity][eid]:
                                valid_ids[related].update(self.G[entity][eid][relation])

            for entity in tqdm(['word', 'brand', 'category', 'related_product'], desc="Final Pruning"):
                print(f"processing: {entity}")
                self.G[entity] = {eid:self.G[entity][eid] for eid in self.G[entity] if eid in valid_ids[entity]}

            removed_entities = self.identify_invalid_ents()
            print(removed_entities)
            logging.info("Graph pruning completed.")
            return self.G
# ==========================================================-================================        
    def remap_entity_ids(self, entity_related_type_info):
        def is_ordered(lst):
            return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

        # Ensure the user and product IDs are ordered
        assert is_ordered(list(self.G['user'].keys())), "User IDs are not ordered"
        user_old_to_new_id = {old: new for new, old in enumerate(self.G['user'].keys())}
        
        assert is_ordered(list(self.G['product'].keys())), "Product IDs are not ordered"
        product_old_to_new_id = {old: new for new, old in enumerate(self.G['product'].keys())}

        # Function to update entity IDs and relationships in the graph
        def update_graph_ids(entity_type, user_id_map, product_id_map):
            new_G = {}

            # Update the entity IDs for users and products
            id_map = user_id_map if entity_type == 'user' else product_id_map if entity_type == 'product' else None
            if id_map:
                for old_id, entity_data in self.G[entity_type].items():
                    new_id = id_map[old_id]
                    new_G[new_id] = entity_data
            else:
                new_G = self.G[entity_type]

            # Update relationships in the graph using the appropriate ID maps
            for eid, entity_data in new_G.items():
                related_entities = entity_related_type_info.get(entity_type, set())
                if 'product' in related_entities:
                    relations = self.reversed_relation_info.get((entity_type, 'product'), [])
                    for relation in relations:
                        if relation in entity_data:
                            entity_data[relation] = [product_id_map.get(rel_id, rel_id) for rel_id in entity_data[relation]]
                
                if 'user' in related_entities:
                    relations = self.reversed_relation_info.get((entity_type, 'user'), [])
                    for relation in relations:
                        if relation in entity_data:
                            entity_data[relation] = [user_id_map.get(rel_id, rel_id) for rel_id in entity_data[relation]]

            self.G[entity_type] = new_G
            logging.info(f"Updated IDs and relations for {entity_type}")

        # Update all entity types in the graph
        for entity_type in self.G.keys():
            update_graph_ids(entity_type, user_old_to_new_id, product_old_to_new_id)

        logging.info("Entity IDs and their relationships have been remapped based on the new ordering.")
        return user_old_to_new_id, product_old_to_new_id

# ==========================================================-================================

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)


    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            assert eh_type in data
            data = data[eh_type]
        if eh_id is not None:
            assert eh_id in data
            data = data[eh_id]
        if relation is not None:
            data = data[relation] if relation in data else []
        return data


    def sample_noise_path(self, metapath, user_id):
        path = [(None, USER, user_id)]
        for i in range(1, len(metapath)):
            next_relation, next_entity = metapath[i]
            vocab_size = len(self.G[next_entity])
            eid = random.choice(range(vocab_size))
            path.append((next_relation, next_entity, eid))
        return path


    def sample_paths(self, metapath, user_id, sample_sizes):
        """BFS with random sampling."""
        paths = [[(None, USER, user_id)]]
        for i in range(1, len(metapath)):
            next_relation, next_entity = metapath[i]
            tmp_paths = []
            for path in paths:
                _, last_entity, last_id = path[-1]
                next_ids = self.get(last_entity, last_id, next_relation)
                if len(next_ids) > sample_sizes[i - 1]:
                    next_ids = np.random.choice(next_ids, size=sample_sizes[i - 1], replace=False)
                for next_id in next_ids:
                    tmp_path = path + [(next_relation, next_entity, next_id)]
                    tmp_paths.append(tmp_path)
            paths = tmp_paths
        paths = [tuple(p) for p in paths]
        return paths


    def sample_paths_with_target(self, metapath, user_id, target_id, num_paths):
        """BFS from both sides."""
        path_len = len(metapath) - 1
        mid_level = int((path_len + 0) / 2)

        # Forward BFS.
        # t1 = time.time()
        forward_paths = [[(None, USER, user_id)]]
        for i in range(1, mid_level + 1):
            next_relation, next_entity = metapath[i]
            tmp_paths = []
            for fp in forward_paths:
                _, last_entity, last_id = fp[-1]
                next_ids = self.get(last_entity, last_id, next_relation)
                for next_id in next_ids:
                    tmp_path = fp + [(next_relation, next_entity, next_id)]
                    tmp_paths.append(tmp_path)
            forward_paths = tmp_paths

        # t2 = time.time()
        # print('---- FP ----')
        # print((t2-t1), len(forward_paths))
        # for i in range(len(forward_paths)):
        #    print(forward_paths[i])

        def _rev_rel(rel):
            if rel.startswith(REV_PREFIX):
                return rel[len(REV_PREFIX) :]
            return REV_PREFIX + rel

        # Backward BFS.
        # t1 = time.time()
        backward_paths = [[(metapath[-1][0], metapath[-1][1], target_id)]]
        for i in reversed(range(mid_level, path_len)):
            curr_relation, curr_entity = metapath[i]
            tmp_paths = []
            for bp in backward_paths:
                next_relation, next_entity, next_id = bp[0]
                curr_ids = self.get(next_entity, next_id, _rev_rel(next_relation))
                for curr_id in curr_ids:
                    tmp_path = [(curr_relation, curr_entity, curr_id)] + bp
                    tmp_paths.append(tmp_path)
            backward_paths = tmp_paths
        # t2 = time.time()
        # print('---- BP ----')
        # print((t2-t1), len(backward_paths))
        # for i in range(len(backward_paths)):
        #    print(backward_paths[i])

        # Find intersection paths.
        final_paths = []
        backward_map = {}
        for bp in backward_paths:
            backward_map[bp[0]] = bp[1:]
        for fp in forward_paths:
            if fp[-1] in backward_map:
                final_paths.append(tuple(fp + backward_map[fp[-1]]))
        if len(final_paths) > num_paths:
            # final_paths = np.random.choice(final_paths, size=num_paths)
            final_paths = random.sample(final_paths, num_paths)
        return final_paths


    def fast_sample_path_with_target(self, mpath_id, user_id, target_id, num_paths, sample_size=100):
        """Sample one path given source and target, using BFS from both sides.
        Returns:
            list of entity ids.
        """
        metapath = self.metapaths[mpath_id]
        path_len = len(metapath) - 1
        mid_level = int((path_len + 0) / 2)

        # Forward BFS (e.g. u--e1--e2--e3).
        forward_paths = [[user_id]]
        for i in range(1, mid_level + 1):
            _, last_entity = metapath[i - 1]
            next_relation, _ = metapath[i]
            tmp_paths = []
            for fp in forward_paths:
                next_ids = self.get(last_entity, fp[-1], next_relation)
                # Random sample ids
                if len(next_ids) > sample_size:
                    # next_ids = np.random.permutation(next_ids)[:sample_size]
                    next_ids = np.random.choice(next_ids, size=sample_size, replace=False)
                for next_id in next_ids:
                    tmp_paths.append(fp + [next_id])
            forward_paths = tmp_paths

        def _rev_rel(rel):
            if rel.startswith(REV_PREFIX):
                return rel[len(REV_PREFIX) :]
            return REV_PREFIX + rel

        # Backward BFS (e.g. e4--e5--e6).
        backward_paths = [[target_id]]
        for i in reversed(range(mid_level + 2, path_len + 1)):  # i=l, l-2,..., mid+2
            next_relation, next_entity = metapath[i]
            tmp_paths = []
            for bp in backward_paths:
                # print(next_entity, bp[0], next_relation)
                curr_ids = self.get(next_entity, bp[0], _rev_rel(next_relation))
                # Random sample ids
                if len(curr_ids) > sample_size:
                    # curr_ids = np.random.permutation(curr_ids)[:sample_size]
                    curr_ids = np.random.choice(curr_ids, size=sample_size, replace=False)
                for curr_id in curr_ids:
                    tmp_paths.append([curr_id] + bp)
            backward_paths = tmp_paths

        # Build hash map for indexing backward paths.
        # e.g. a dict with key=e3 and value=(e4--e5--e6).
        backward_map = {}
        next_relation, next_entity = metapath[mid_level + 1]
        for bp in backward_paths:
            curr_ids = self.get(next_entity, bp[0], _rev_rel(next_relation))
            if len(curr_ids) > sample_size:
                curr_ids = np.random.choice(curr_ids, size=sample_size, replace=False)
            for curr_id in curr_ids:
                if curr_id not in backward_map:
                    backward_map[curr_id] = []
                backward_map[curr_id].append(bp)

        # Find intersection of forward paths and backward paths.
        final_paths = []
        for fp_idx in np.random.permutation(len(forward_paths)):
            fp = forward_paths[fp_idx]
            mid_id = fp[-1]
            if mid_id not in backward_map:
                continue
            np.random.shuffle(backward_map[mid_id])
            for bp in backward_map[mid_id]:
                final_paths.append(fp + bp)
                if len(final_paths) >= num_paths:
                    break
            if len(final_paths) >= num_paths:
                break
        return final_paths


    def count_paths_with_target(self, mpath_id, user_id, target_id, sample_size=50):
        """This is an approx count, not exact."""
        metapath = self.metapaths[mpath_id]
        path_len = len(metapath) - 1
        mid_level = int((path_len + 0) / 2)

        # Forward BFS (e.g. u--e1--e2--e3).
        forward_ids = [user_id]
        for i in range(1, mid_level + 1):  # i=1, 2,..., mid
            _, last_entity = metapath[i - 1]
            next_relation, _ = metapath[i]
            tmp_ids = []
            for eid in forward_ids:
                next_ids = self.get(last_entity, eid, next_relation)
                if len(next_ids) > sample_size:
                    next_ids = np.random.choice(next_ids, size=sample_size, replace=False).tolist()
                tmp_ids.extend(next_ids)
            forward_ids = tmp_ids
        # cnt = len([_ for i in forward_ids if i == target_id])
        # return cnt
        # forward_ids, forward_counts = np.unique(forward_ids, return_counts=True)
        # print(forward_ids, forward_counts)

        def _rev_rel(rel):
            if rel.startswith(REV_PREFIX):
                return rel[len(REV_PREFIX) :]
            return REV_PREFIX + rel

        # Backward BFS (e.g. e4--e5--e6).
        backward_ids = [target_id]
        for i in reversed(range(mid_level + 1, path_len + 1)):  # i=l, l-1,..., mid+1
            next_relation, next_entity = metapath[i]
            tmp_ids = []
            for eid in backward_ids:
                curr_ids = self.get(next_entity, eid, _rev_rel(next_relation))
                tmp_ids.extend(curr_ids)
            backward_ids = tmp_ids
        # backward_ids, backward_counts = np.unique(backward_ids, return_counts=True)
        # print(backward_ids, backward_counts)

        count = len(set(forward_ids).intersection(backward_ids))
        return count


    def is_valid_path(self, mpid, path):
        metapath = self.metapaths[mpid]
        if len(metapath) != len(path):
            return False
        graph = self.get(metapath[0][1])
        if path[0] not in graph:
            return False
        last_node = (metapath[0][1], path[0])
        for i in range(1, len(path)):
            relation, entity = metapath[i]
            node_id = path[i]
            candidates = self.get(last_node[0], last_node[1], relation)
            if node_id not in candidates:
                return False
            last_node = (entity, node_id)
        return True


    def shrink_labels(self, uids,dataset='beauty'):
        train_labels = utils.load_labels(dataset, mode='train')
        shrunk_labels = {u: train_labels[u] for u in uids if u in train_labels}
        utils.save_labels(shrunk_labels, dataset, mode='train')

        test_labels = utils.load_labels(dataset, mode='train')
        shrunk_labels = {u: train_labels[u] for u in uids if u in test_labels}
        utils.save_labels(shrunk_labels, dataset, mode='test')

    def update_labels_ids(self):
        train_labels = utils.load_labels('beauty', mode='train')
        train_labels = {self.user_old_to_new_id[uid]:[self.product_old_to_new_id[pid] for pid in pid_l] for uid, pid_l in train_labels.items()}
        utils.save_labels(train_labels, 'beauty', mode='train')

        test_labels = utils.load_labels('beauty', mode='test')
        test_labels = {self.user_old_to_new_id[uid]:[self.product_old_to_new_id[pid] for pid in pid_l] for uid, pid_l in test_labels.items()}
        utils.save_labels(test_labels, 'beauty', mode='test')