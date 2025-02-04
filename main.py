import CF_utils  
from recommender import get_recommendation
from pprint import pprint
from counterfactual_analysis import CFAnalyzer

from data_registry import DataRegistry


def main(args):
    DataRegistry.initialize(args)
    print("Getting recommendations for User...")
    user_recommendations = get_recommendation(args)
    print("Recommendations paths:")
    pprint(user_recommendations)
    
    print("Running counterfactual analysis...")
    # recommendation_to_analyze = user_recommendations[-2]  # Select the path to analyze
    DataRegistry.populate_attributes(
        recommendations=user_recommendations,
        recommendation_to_analyze=user_recommendations[-1],
        # recommendation_to_analyze=user_recommendations[-1],
        # TODO: hard-coded
        k=5,
        args=args
    )


    # Initialize and call the CFAnalyzer
    # TODO:
    cf_analyzer = CFAnalyzer(user_recommendations, DataRegistry.recommendation_to_analyze, args)
    cf_analyzer.analyze()  # Run the counterfactual analysis using the __call__ method

if __name__ == '__main__':
    args = CF_utils.parse_args()
    main(args)

