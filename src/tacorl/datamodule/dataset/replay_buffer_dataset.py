import numpy as np
from torch.utils.data import Dataset

from tacorl.utils.path import get_file_list
from tacorl.utils.transforms import TransformManager


class ReplayBufferDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        modalities: list,
        transform_manager: TransformManager = None,
        transf_type: str = "train",
        include_next_obs: bool = False,
        device: str = "cpu",
        *args,
        **kwargs
    ):
        """
        Args
            data_dir: path of the dir where the data is stored
            modalities: list of strings with the keys of the modalities that
                        will be used to train the network
        """
        self.modalities = modalities
        self.transform_manager = transform_manager
        self.transf_type = transf_type
        self.file_list = get_file_list(data_dir)
        self.file_list = sorted(
            self.file_list, key=lambda file: int(file.stem.split("_")[-1])
        )
        self.include_next_obs = include_next_obs
        self.device = device

    def __len__(self):
        return len(self.file_list)

    def get_transf_obs(self, obs: dict, action: np.array = None):
        if action is not None:
            obs["rel_actions"] = action
        transf_obs = {
            modality: obs[modality] for modality in self.modalities if modality in obs
        }
        transf_obs = self.transform_manager(
            transf_obs, self.transf_type, device=self.device
        )
        return transf_obs

    def __getitem__(self, idx):
        item = {}
        data = np.load(self.file_list[idx], allow_pickle=True)
        if self.include_next_obs:
            item["obs"] = self.get_transf_obs(
                data["state"].item(), action=data["action"]
            )
            item["next_obs"] = self.get_transf_obs(data["next_state"].item())
        else:
            item = self.get_transf_obs(data["state"].item(), action=data["action"])
        return item
