import random
import psutil
import utils
import shutil
import os
from itertools import chain

def print_summary(summary):
    for entity, report in summary.items():
        print(entity)
        for report_item, value in report.items():
            if isinstance(value, dict):
                # If the value is a dictionary, iterate over its items
                print(f"{report_item}:")
                for key, num in value.items():
                    # Format the numbers within the dictionary with commas
                    print(" " * 8 + f"{key}: {num:,}")
            else:
                # If the value is not a dictionary, format it directly with commas
                print(f"{report_item}: {value:,}")
        print("-" * 50)


def summarize_kg(G):
    summary = {}
    total_entities = 0
    total_relations = 0

    for entity_type, entities in G.items():
        num_entities = len(entities)  # Number of entities of this type
        num_relations = {}  # Initialize relation count dictionary

        # Iterate over each entity and its relations
        for entity_id, relations in entities.items():
            for relation, related_entities in relations.items():
                if relation not in num_relations:
                    num_relations[relation] = 0  # Initialize if this relation has not been encountered yet
                num_relations[relation] += len(related_entities)  # Count each relation's entities

        summary[entity_type] = {'Number of Entities': num_entities, 'Number of Relations': num_relations}
        total_entities += num_entities
        total_relations += sum(num_relations.values())  # Sum up all relation entities for the type

    summary['Total'] = {'Total Entities': total_entities, 'Total Relations': total_relations}
    return summary

def compare_kgs(original_kg, sampled_kg):
    original_summary = summarize_kg(original_kg)
    sampled_summary = summarize_kg(sampled_kg)
    comparison_rates = {}

    for entity_type, data in original_summary.items():
        if entity_type != 'Total' and entity_type in sampled_summary:
            entity_rate = (sampled_summary[entity_type]['Number of Entities'] / data['Number of Entities']) * 100
            relation_rates = {}
            for relation, count in data['Number of Relations'].items():
                if relation in sampled_summary[entity_type]['Number of Relations']:
                    relation_rates[relation] = (sampled_summary[entity_type]['Number of Relations'][relation] / count) * 100
                else:
                    relation_rates[relation] = 0.0
            comparison_rates[entity_type] = {'Entity Coverage %': entity_rate, 'Relation Coverage %': relation_rates}

    # Comparing totals
    total_entities_rate = (sampled_summary['Total']['Total Entities'] / original_summary['Total']['Total Entities']) * 100
    total_relations_rate = (sampled_summary['Total']['Total Relations'] / original_summary['Total']['Total Relations']) * 100
    comparison_rates['Total'] = {'Total Entity Coverage %': total_entities_rate, 'Total Relation Coverage %': total_relations_rate}
    for key, value in comparison_rates.items():
        print(f"Coverage for {key}: {value}")
    return comparison_rates

def sample_users_and_products(user_sample_size):
    # Load the training labels, which contain user-product mappings
    train_labels = utils.load_labels('beauty', 'train')
    test_labels = utils.load_labels('beauty', 'test')
    random.seed(42)

    # Randomly sample user IDs based on the user_sample_size
    sampled_user_ids = random.sample(list(train_labels.keys()), user_sample_size)

    # Collect product IDs for the sampled users
    product_ids = set()
    for user_id in sampled_user_ids:
        product_ids.update(train_labels[user_id])  # Add all products associated with the user to the set
        product_ids.update(test_labels[user_id])  # Add all products associated with the user to the set

    # Create a dictionary to store the sampled entities
    sampled_entities = {}
    sampled_entities['user'] = sampled_user_ids
    sampled_entities['product'] = list(product_ids)  # Convert set to list to have a list of product IDs


    return sampled_entities

def check_memory_limit(max_memory_MB):
    process = psutil.Process()
    mem_info = process.memory_info()
    usage = mem_info.rss / 1024 / 1024  # Convert bytes to MB
    print(f"Current memory usage: {usage:.2f} MB")
    if usage > max_memory_MB:
        print("Memory limit exceeded, stopping execution.")
        raise MemoryError("Memory limit exceeded")

def filter_allowed_entities(entities, type_key, entity_sample_limits):
    if type_key in entity_sample_limits:
        allowed_ids = set(entity_sample_limits[type_key])
        return set(entities).intersection(allowed_ids)
    return entities

def shrink_graph(G, relation_info, entity_order, entity_ids, max_depth, max_memory_MB=2000, sample_rate=0.1):
    depth = 0
    queue = [entity_order[0]]
    processed_types = set()
    sampled_graph = {}
    entity_sample_limits = {
        'user': entity_ids['user'],  # Assuming entity_ids['user'] is a list of user IDs
        'product': entity_ids['product']  # Assuming entity_ids['product'] is a list of product IDs
    }

    while queue and depth < max_depth:
        current_type = queue.pop(0)
        if current_type not in processed_types:
            processed_types.add(current_type)

            if current_type in entity_ids and current_type not in sampled_graph:
                sampled_ids = list(filter_allowed_entities(entity_ids[current_type], current_type, entity_sample_limits))
            else:
                sampled_ids = list(sampled_graph.get(current_type, {}).keys())

            sampled_graph[current_type] = {eid: G[current_type][eid] for eid in sampled_ids if eid in G[current_type]}

            for entity_id in list(sampled_graph[current_type].keys()):
                if entity_id in G[current_type]:
                    for relation, related_entities in G[current_type][entity_id].items():
                        related_type = relation_info[relation][1]
                        valid_entities = filter_allowed_entities(related_entities, related_type, entity_sample_limits)
                        if valid_entities:
                            if relation not in sampled_graph[current_type][entity_id]:
                                sampled_graph[current_type][entity_id][relation] = []
                            sampled_graph[current_type][entity_id][relation] = [eid for eid in sampled_graph[current_type][entity_id][relation] if eid in valid_entities]

            check_memory_limit(max_memory_MB)

            next_entities = {}
            for entity_id in sampled_ids:
                for relation, related_entities in G[current_type][entity_id].items():
                    related_type = relation_info[relation][1]
                    if related_type not in next_entities:
                        next_entities[related_type] = set()

                    filtered_related_entities = filter_allowed_entities(related_entities, related_type, entity_sample_limits)
                    next_entities[related_type].update(filtered_related_entities)

            for next_type, ids in next_entities.items():
                if next_type not in processed_types:
                    queue.append(next_type)
                if next_type not in sampled_graph:
                    sampled_graph[next_type] = {}

                for nid in ids:
                    if nid in G[next_type]:
                        sampled_graph[next_type][nid] = G[next_type][nid]

        if not queue or current_type == entity_order[-1]:
            depth += 1
            queue.extend([t for t in entity_order if t not in processed_types and t in next_entities])

    print("Finished processing all entities.")
    print()
    print("*" * 10 + "Sampled KG Report" + "*" * 10)
    print_summary(summarize_kg(sampled_graph))
    print("*" * 10 + "Coverage Report" + "*" * 10)
    compare_kgs(G, sampled_graph)

    return sampled_graph

def shrink_labels(dataset, uids):

    train_labels = utils.load_labels(dataset, mode='train')
    shrunk_labels = {u: train_labels[u] for u in uids if u in train_labels}
    utils.save_labels(shrunk_labels, dataset, mode='train')

    test_labels = utils.load_labels(dataset, mode='train')
    shrunk_labels = {u: train_labels[u] for u in uids if u in test_labels}
    utils.save_labels(shrunk_labels, dataset, mode='test')

def shrink_kg_object(kg_object, entity_order = ['user', 'product', 'word', 'brand', 'category', 'related_product'], 
                                sample=1, max_depth=3, max_memory_MB=2000):
    total_users = 22363
    entity_ids = sample_users_and_products(sample)
    samppled_kg = shrink_graph(kg_object.G, kg_object.relation_info, entity_order=entity_order, entity_ids=entity_ids, max_depth=max_depth, max_memory_MB=max_memory_MB)
    kg_object.G = samppled_kg
    
    #TODO: What about the product ids match
    shrink_labels('beauty', list(kg_object.G['user'].keys()))

def summarize_data_interactions(sampled_kg):
    # Load labels
    train_labels = utils.load_labels('beauty', 'train')
    test_labels = utils.load_labels('beauty', 'test')
    
    # Extract user and product IDs from training data
    train_uids = set(train_labels.keys())
    train_pids = set(chain.from_iterable(train_labels.values()))
    
    # Extract user and product IDs from testing data
    test_uids = set(test_labels.keys())
    test_pids = set(chain.from_iterable(test_labels.values()))
    
    # Extract user and product IDs from knowledge graph
    kg_users = set(sampled_kg['user'].keys())
    kg_products = set(sampled_kg['product'].keys())

    # Compute differences and print summaries
    print("Training User IDs not in KG:", train_uids.difference(kg_users))
    print("KG User IDs not in Training:", len(kg_users.difference(train_uids)))
    print("Training Product IDs not in KG:", train_pids.difference(kg_products))
    print("Training Product IDs not in KG count:", len(train_pids.difference(kg_products)))
    print("KG Product IDs not in Training:", len(kg_products.difference(train_pids)))
    
    print("Testing User IDs not in KG:", test_uids.difference(kg_users))
    print("KG User IDs not in Testing:", len(kg_users.difference(test_uids)))
    print("Testing Product IDs not in KG:", test_pids.difference(kg_products))
    print("Testing Product IDs not in KG count:", len(test_pids.difference(kg_products)))
    print("KG Product IDs not in Testing:", len(kg_products.difference(test_pids)))

    # Optionally print a summary of the knowledge graph
    # if 'print_summary' in globals() and callable(print_summary):
    #     print_summary(summarize_kg(sampled_kg))





