from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

import tacorl


def get_file_list(data_dir, extension=".npz", sort_list=False):
    """retrieve a list of files inside a folder"""
    dir_path = Path(data_dir)
    dir_path = dir_path.expanduser()
    assert dir_path.is_dir(), f"{data_dir} is not a valid dir path"
    file_list = []
    for x in dir_path.iterdir():
        if x.is_file() and extension in x.suffix:
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(get_file_list(x, extension))
    if sort_list:
        file_list = sorted(file_list, key=lambda file: file.name)
    return file_list


def get_cwd():
    if HydraConfig.initialized():
        cwd = Path(get_original_cwd())
    else:
        cwd = Path.cwd()
    return cwd


def add_cwd(path):
    return str((get_cwd() / path).resolve())


def pkg_path(rel_path):
    """Generates a global path that is relative to
    the root of thesis package.
    (Could be generalized for any python module)

    Args:
        rel_path (str): Relative path within drlfads package

    Returns:
        str: Global path.
    """
    return str(Path(tacorl.__path__[0], rel_path))
