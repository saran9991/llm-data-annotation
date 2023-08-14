import os

def get_next_version(directory, prefix='annotated_'):
    existing_files = os.listdir(directory)
    versions = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.startswith(prefix)]
    if versions:
        return max(versions) + 1
    else:
        return 1