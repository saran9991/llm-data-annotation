import os

def get_next_version(directory: str, prefix: str = 'annotated_') -> int:
    """
    Get the next available version number for a file with a given prefix in a directory.

    Parameters:
    - directory (str): Directory path where the files are located.
    - prefix (str): Prefix for the filename (default is 'annotated_').

    Returns:
    - int: The next available version number.
    """
    existing_files = os.listdir(directory)
    versions = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.startswith(prefix)]
    if versions:
        return max(versions) + 1
    else:
        return 1