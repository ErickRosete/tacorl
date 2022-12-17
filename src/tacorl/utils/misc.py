import importlib
import logging
import os
from typing import List, Union

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist

log_print = logging.getLogger(__name__)


def map_value_to_numpy(value):
    if isinstance(value, list):
        return np.asarray(value)
    elif isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    else:
        return value


def get_distances(emb1, emb2):
    """Get L2 norm distance between two embeddings"""

    assert np.shape(emb1) == np.shape(
        emb2
    ), "dist shape not same, " "emb1.shape{} and emb1.shape {}".format(
        emb1.shape, emb2.shape
    )

    return np.linalg.norm(emb1 - emb2)


def dict_to_list_of_dicts(input: dict, to_numpy=False) -> list:
    """
    Transform a dict to a list of dicts
    e.g.
        input
            dict = {'a': [0, 1], 'b': [2, 3]}
        return
            [{'a': 0, 'b': 2},
             {'a': 1, 'b': 3}]
    """
    output = []
    for key, array in input.items():
        for i, value in enumerate(array):
            if len(output) <= i:
                output.append({})
            if to_numpy:
                output[i][key] = map_value_to_numpy(value)
            else:
                output[i][key] = value
    return output


def list_of_dicts_to_dict_tensor(input: list) -> dict:
    """
    Transform a list of dicts to a dict with tensors
    e.g.
        input
            [{'a': 0, 'b': 2},
             {'a': 1, 'b': 3}]
        return
             {'a': tensor([0, 1]), 'b': tensor([2, 3])}
    """
    output = list_of_dicts_to_dict(input)
    for key, value in output.items():
        if torch.is_tensor(value[0]):
            output[key] = torch.stack(value, dim=0)
        else:
            output[key] = torch.tensor(output[key])
    return output


def list_of_dicts_to_dict(input: list, to_numpy=False) -> dict:
    """
    Transform a list of dicts to a dict
    e.g.
        input
            [{'a': 0, 'b': 2},
             {'a': 1, 'b': 3}]
        return
             {'a': [0, 1], 'b': [2, 3]}
    """
    output = {}
    for subdict in input:
        for key, value in subdict.items():
            if key not in output:
                output[key] = [value]
            else:
                output[key].append(value)

    if to_numpy:
        for key in output.keys():
            output[key] = map_value_to_numpy(output[key])
    return output


def flatten(t: List[List]):
    return [item for sublist in t for item in sublist]


def flatten_list_of_dicts(t: List[dict]):
    return {k: v for d in t for k, v in d.items()}


def transform_to_list(batch, to_numpy=False):
    if isinstance(batch, dict):
        return dict_to_list_of_dicts(batch, to_numpy)

    new_list = []
    for element in batch:
        if to_numpy:
            new_list.append(map_value_to_numpy(element))
        else:
            new_list.append(element)
    return new_list


def filter_obs(obs: Union[dict, torch.Tensor], keep_obs: torch.BoolTensor):
    if isinstance(obs, dict):
        filtered_obs = {}
        for key in obs.keys():
            filtered_obs[key] = filter_obs(obs[key], keep_obs)
        return filtered_obs
    else:
        return obs[keep_obs]


def expand_array(obs, n_samples, reshape: bool = True):
    expanded_obs = obs.expand(n_samples, *obs.shape)  # n, bs, ...
    if reshape:
        return expanded_obs.reshape(-1, *obs.shape[1:])  # n * bs, ...
    else:
        return expanded_obs


def expand_dict(obs, n_samples, reshape: bool = True):
    expanded_obs = {}
    for key in obs.keys():
        if isinstance(obs[key], dict):
            expanded_obs[key] = expand_dict(obs[key], n_samples, reshape=reshape)
        else:
            expanded_obs[key] = expand_array(obs[key], n_samples, reshape=reshape)
    return expanded_obs


def expand_obs(obs, n_samples, reshape: bool = True):
    if isinstance(obs, dict):
        return expand_dict(obs, n_samples, reshape=reshape)
    return expand_array(obs, n_samples, reshape=reshape)


def get_obs_device(obs):
    if isinstance(obs, dict):
        return get_obs_device(list(obs.values())[0])
    else:
        return obs.device


def extract_img_from_obs(obs):
    """Extract image according to ordered modalities priority"""
    if isinstance(obs, dict) and "observation" in obs:
        return extract_img_from_obs(obs["observation"])
    ordered_modalities = ["rgb_static", "depth_static", "rgb_gripper", "depth_gripper"]
    if isinstance(obs, dict):
        for modality in ordered_modalities:
            if modality in obs:
                return obs[modality]
    return None


def add_img_thumbnail_to_imgs(imgs: np.ndarray, img_thumbnail: np.ndarray):
    size = imgs.shape[-2:]
    i_h = int(size[0] / 3)
    i_w = int(size[1] / 3)
    resize_goal_img = cv2.resize(
        img_thumbnail, dsize=(i_h, i_w), interpolation=cv2.INTER_CUBIC
    )
    mod_goal_img = np.expand_dims(resize_goal_img.transpose(2, 0, 1), axis=0)
    imgs[..., -i_h:, :i_w] = mod_goal_img
    return imgs


def add_text(img, lang_text):
    height, width, _ = img.shape
    if lang_text != "":
        coord = (1, int(height - 10))
        font_scale = (0.7 / 500) * width
        thickness = 1
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=thickness * 3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


# PytorchLightning logger help functions
def pl_log_metrics(
    metrics: dict = {},
    pl_module: pl.LightningModule = None,
    log_type: str = "validation",
    on_epoch: bool = True,
    on_step: bool = True,
    sync_dist: bool = False,
):
    """log all dictionary values in pytorch lightning"""
    for key in metrics.keys():
        pl_module.log(
            name="%s/%s" % (log_type, key),
            value=metrics[key],
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
        )


def empty_cache():
    """
    Clear GPU reserved memory. Do not call this unnecessarily.
    """
    mem1 = torch.cuda.memory_reserved(dist.get_rank())
    torch.cuda.empty_cache()
    mem2 = torch.cuda.memory_reserved(dist.get_rank())
    log = logging.getLogger(__name__)
    log.info(
        f"GPU: {dist.get_rank()} freed {(mem1 - mem2) / 10**9:.1f}GB of reserved memory"
    )
    # log.info(torch.cuda.memory_summary(dist.get_rank()))


def delete_file(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def log_rank_0(*args, **kwargs):
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    log_print.info(*args, **kwargs)


def load_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_portion_of_batch_ids(percentage: float, batch_size: int) -> np.ndarray:
    """
    Select percentage * batch_size indices spread out evenly throughout array
    Examples
    ________
     >>> get_portion_of_batch_ids(percentage=0.5, batch_size=32)
     array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
     >>> get_portion_of_batch_ids(percentage=0.2, batch_size=32)
     array([ 0,  5, 10, 16, 21, 26])
     >>> get_portion_of_batch_ids(percentage=0.01, batch_size=64)
     array([], dtype=int64)
    """
    num = int(batch_size * percentage)
    if num == 0:
        return np.array([], dtype=np.int64)
    indices = np.arange(num).astype(float)
    stretch = batch_size / num
    indices *= stretch
    return np.unique(indices.astype(np.int64))


def log_sum_exp(x):
    """numerically stable log_sum_exp implementation that prevents overflow"""
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5 * torch.log(one_plus_x / one_minus_x)
