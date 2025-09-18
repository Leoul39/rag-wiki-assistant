from pathlib import Path
import os
from typing import Union
import yaml

DATA_DIR = "data"  # folder containing the .txt files

def load_text_file(file_stem: str, data_dir: str = DATA_DIR) -> str:
    """
    Loads a single text file by stem (filename without extension).
    """
    file_path = Path(os.path.join(data_dir, f"{file_stem}.txt"))
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except IOError as e:
        raise IOError(f"Error reading text file: {e}") from e


def load_all_text_files(data_dir: str = DATA_DIR) -> list[str]:
    """
    Loads all .txt files from a directory and returns their contents as a list.
    """
    texts = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            stem = Path(fname).stem
            texts.append(load_text_file(stem, data_dir))
    return texts

def load_yaml_config(file_path: Union[str, Path]) -> dict:
    """Loads a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there's an error parsing YAML.
        IOError: If there's an error reading the file.
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    # Read and parse the YAML file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e
# if __name__=='__main__':
#     files = load_all_text_files()
#     print(files)