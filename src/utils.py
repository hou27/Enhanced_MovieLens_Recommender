import os

def get_folder_path(**kwargs):
    path = os.path.join('results', kwargs.get('model', ''), kwargs.get('dataset', ''))
    os.makedirs(path, exist_ok=True)
    return path