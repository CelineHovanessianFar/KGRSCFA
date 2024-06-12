import shutil
import os

def get_base_path():
    current_path = os.getcwd()
    if 'CAFE' in current_path:
        return current_path.split('CAFE')[0]  # Returns the directory just before the 'CAFE' folder
    return current_path  # Returns the current path if not inside CAFE

def clear_directories():
    base_path = get_base_path()
    directories_to_delete = [os.path.join(base_path, 'CAFE', 'tmp'), os.path.join(base_path, 'tmp')]
    
    for directory in directories_to_delete:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Deleted {directory}")
        else:
            print(f"Directory {directory} does not exist")

def replace_labels_files():
    base_path = get_base_path()
    files_to_replace = ['train_labels.txt.gz', 'test_labels.txt.gz']  # List of files to replace

    for file_name in files_to_replace:
        source_path = os.path.join(base_path, 'CAFE', 'data', 'Beauty', 'original', file_name)
        destination_path = os.path.join(base_path, 'CAFE', 'data', 'Beauty', file_name)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
            print(f"Replaced {destination_path} with original")
        else:
            print(f"Original file {source_path} does not exist")


def reset_neighbors():
    files_to_delete = [
        r'tmp\neighbor_entities_849.pkl',
        r'tmp\raw_counter_scores_0_849.pkl'
    ]
    
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")



# clear_directories()
# replace_labels_files()
reset_neighbors()