import sys
sys.path.append('CAFE')

from CAFE import utils
from kg_info import KGENtitiesRelationsInfo


class DataRegistry:
    # Public class attributes
    kg = None
    kg_info = None
    recommendations = None
    recommendation_to_analyze = None
    dataset = None
    k = None

    @classmethod
    def initialize(cls, args):
        if cls.kg is None:
            cls.kg = utils.load_kg(args.dataset)
        if cls.kg_info is None:
            cls.initialize_kg_info()
    
    @classmethod
    def populate_attributes(cls, recommendations, recommendation_to_analyze, k, args):
        cls.recommendations = recommendations
        cls.recommendation_to_analyze = recommendation_to_analyze
        cls.k = k

    @classmethod
    def initialize_kg_info(cls):
        cls.kg_info = KGENtitiesRelationsInfo()