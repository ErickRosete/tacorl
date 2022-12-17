import logging
import os

import torch
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id

logger = logging.getLogger(__name__)


def set_current_egl_device():
    device = torch.cuda.current_device()
    device = torch.device(device)
    set_egl_device(device)


def set_all_egl_devices():
    """Sets all cuda devices as EGL Visible Devices"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        cuda_ids = [int(cuda_id) for cuda_id in cuda_ids]
        egl_ids = []
        for cuda_id in cuda_ids:
            try:
                # egl_id = get_egl_device_id(cuda_id)
                egl_id = 0
                logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")
                egl_ids.append(egl_id)
            except EglDeviceNotFoundError:
                logger.warning("Couldn't find correct EGL device for cuda:%d" % cuda_id)
        os.environ["EGL_VISIBLE_DEVICES"] = ",".join(str(egl_id) for egl_id in egl_ids)
    else:
        set_current_egl_device()


def set_egl_device(device):
    cuda_id = device.index if device.type == "cuda" else 0
    try:
        egl_id = get_egl_device_id(cuda_id)
    except EglDeviceNotFoundError:
        logger.warning(
            "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0."
        )
        egl_id = 0
    os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")
