import os
import subprocess
import CF_utils  
from recommender import get_recommendation
from pprint import pprint
from counterfactual_analysis import CounterfactualFramework

def is_model_trained(model_path):
    return os.path.exists(model_path)

def preprocess_completed():
    count_file = 'CAFE/tmp/Beauty/path_count.pkl'
    return os.path.exists(count_file)

def main(args):
    
    if not preprocess_completed():
        print("Running preprocess.py...")
        subprocess.run(["python", args.preprocess_dir], check=True)
    print('Preprocess Step Already Completed!')
        
    if not is_model_trained(args.symbolic_model):
        # Run train_neural_symbol.py
        print("Running train_neural_symbol.py...")
        subprocess.run(["python", args.train_neural_symbol_dir], check=True)
    print("Recommender System already trained.")
    
    print("Getting recommendations for User...")
    user_recommendations = get_recommendation(args)
    print("Recommendations paths:")
    pprint(user_recommendations)
    
    print("Running counterfactual analysis...")
    print('#' * 100)
    path_to_analyize = user_recommendations[0]
    print(user_recommendations[0])
    print('#' * 100)
    cf_framework = CounterfactualFramework(args, user_recommendations, path_to_analyize)
    cf_framework.counterfactual_explanation_textual()


if __name__ == '__main__':
    args = CF_utils.parse_args()
    main(args)
