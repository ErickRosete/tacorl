from pathlib import Path
from typing import Union

import hydra
import torch
from omegaconf import OmegaConf

from tacorl.utils.misc import load_class


def get_shape_from_dict(input: dict):
    new_input = list(input.values())[0]
    if isinstance(new_input, dict):
        return get_shape_from_dict(new_input)
    return new_input.shape


def get_batch_size_from_input(input: torch.Tensor):
    """Extract batch size from an input"""
    if isinstance(input, dict):
        input_shape = get_shape_from_dict(input)
    else:
        input_shape = input.shape

    # TODO: change condition
    if len(input_shape) == 1 or len(input_shape) == 3:
        return 1
    else:
        return input_shape[0]


def transform_to_tensor(x, dtype=torch.float, grad=True, device="cuda"):
    if isinstance(x, dict):
        tensor = {
            k: torch.tensor(v, dtype=dtype, device=device, requires_grad=grad)
            for k, v in x.items()
        }
    else:
        tensor = torch.tensor(
            x, dtype=dtype, device=device, requires_grad=grad
        )  # B,S_D
    return tensor


def transform_to_device(x, device="cuda"):
    if isinstance(x, dict):
        new_x = {k: v.to(device) for k, v in x.items()}
    else:
        new_x = x.to(device)
    return new_x


def calc_out_size(w, h, kernel_size, padding=0, stride=1):
    width = (w - kernel_size + 2 * padding) // stride + 1
    height = (h - kernel_size + 2 * padding) // stride + 1
    return width, height


def set_parameter_requires_grad(model, requires_grad):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = requires_grad


def freeze_params(model):
    set_parameter_requires_grad(model, requires_grad=False)


def load_transform_manager_from_dir(dirpath):
    dir_path = Path(dirpath)
    if dir_path.is_file():
        dir_path = dir_path.parents[0]
    config = get_config_from_dir(dir_path)
    transform_manager_cfg = config.datamodule.transform_manager
    return hydra.utils.instantiate(transform_manager_cfg)


def load_pl_module_from_checkpoint(filepath: Union[Path, str], epoch=-1):
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.is_dir():
        filedir = filepath
        ckpt_path = get_checkpoint_i_from_dir(dir=filedir, i=epoch)
    elif filepath.is_file():
        assert filepath.suffix == ".ckpt", "File must have .ckpt extension"
        ckpt_path = filepath
        filedir = filepath.parents[0]
    else:
        raise ValueError(f"not valid file path: {str(filepath)}")
    config = get_config_from_dir(filedir)
    class_name = config.module.pop("_target_")
    if "_recursive_" in config.module:
        del config.module["_recursive_"]
    print(f"class_name {class_name}")
    module_class = load_class(class_name)
    print(f"Loading model from {ckpt_path}")
    model = module_class.load_from_checkpoint(ckpt_path, **config.module)
    print(f"Finished loading model {ckpt_path}")
    return model


def get_checkpoint_i_from_dir(dir, i: int = -1):
    ckpt_paths = list(dir.rglob("*.ckpt"))
    if i == -1:
        for ckpt_path in ckpt_paths:
            if ckpt_path.stem == "last":
                return ckpt_path

    # Search for ckpt of epoch i
    for ckpt_path in ckpt_paths:
        split_path = str(ckpt_path).split("_")
        for k, word in enumerate(split_path):
            if word == "epoch":
                if int(split_path[k + 1]) == i:
                    return ckpt_path

    sorted(ckpt_paths, key=lambda f: f.stat().st_mtime)
    return ckpt_paths[i]


def get_config_from_dir(dir):
    dir = Path(dir)
    config_yaml = list(dir.rglob("*config.yaml"))[0]
    return OmegaConf.load(config_yaml)


if __name__ == "__main__":
    play_lmp_dir = Path("~/tacorl/models/play_lmp").expanduser()
    model = load_pl_module_from_checkpoint(files_dir=play_lmp_dir)
    print()
