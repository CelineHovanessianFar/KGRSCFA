import logging
import utils

class ShrinkPosprocess:
    def __init__(self, kg_obj) -> None:
        self.kg = kg_obj
        self.G = kg_obj.G
        self.relation_info = self.kg.relation_info
    
    def remap_entity_ids(self):
        def is_ordered(lst):
            return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

        new_id_maps = {}
        old_id_maps = {}

        
        # Generate new ID mappings for each entity type in the graph
        for entity_type in self.G.keys():
            assert is_ordered(list(self.G[entity_type].keys())), f"{entity_type} IDs are not ordered"
            new_id_maps[entity_type] = {old: new for new, old in enumerate(self.G[entity_type].keys())}
            old_id_maps[entity_type] = {new: old for old, new in new_id_maps[entity_type].items() }

        # Function to update entity IDs and relationships in the graph
        def update_graph_ids(entity_type, new_id_map):
            new_G = {}
            
            # Update the entity IDs for this entity type
            for old_id, entity_data in self.G[entity_type].items():
                new_id = new_id_map[old_id]
                new_G[new_id] = entity_data
                
            # Update relationships in the graph using the appropriate ID maps
            for eid, entity_data in new_G.items():
                for relation, related_entities in entity_data.items():
                    target_type = self.relation_info[relation][1]
                    target_id_map = new_id_maps[target_type]
                    entity_data[relation] = [target_id_map[rel_id] for rel_id in related_entities]
            
            self.G[entity_type] = new_G
            logging.info(f"Updated IDs and relations for {entity_type}")

            # Update all entity types in the graph
        for entity_type in self.G.keys():
            update_graph_ids(entity_type, new_id_maps[entity_type])

        logging.info("Entity IDs and their relationships have been remapped based on the new ordering.")
        print('='*60 + '!' * 10 +'='*60)
        self.kg.product_old_to_new_id = new_id_maps['product']
        self.kg.user_old_to_new_id = new_id_maps['user']
        self.new_id_maps = old_id_maps
        self.old_id_maps = new_id_maps


    def update_embeddings(self):
        embeds = utils.load_embed('beauty')
        for entity in self.G:
            valid_ids = list(self.old_id_maps[entity].keys())
            embeds[entity] = embeds[entity][valid_ids]

        for relation, entity_pair in self.relation_info.items():
            if relation.startswith('rev'):
                continue
            target_type = entity_pair[1]
            valid_ids = list(self.old_id_maps[target_type].keys())
            embeds[relation] = (embeds[relation][0],embeds[relation][1][valid_ids])

        utils.save_embed('beauty', embeds)
        
    def update_ids_embeddings(self):
        logging.info("Starting the update of the ids and embeddings...")
        self.remap_entity_ids()
        self.update_embeddings()
        logging.info("The update of the ids and embeddings is finished")

        