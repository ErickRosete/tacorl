import numpy as np
from torch.utils.data import Dataset

from tacorl.utils.path import get_file_list
from tacorl.utils.transforms import TransformManager


class OfflineReplayBufferDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        modalities: list,
        transform_manager: TransformManager = None,
        transf_type: str = "train",
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
        self.device = device

    def __len__(self):
        return len(self.file_list)

    def get_transf_obs(self, obs: dict):
        transf_obs = {
            modality: obs[modality] for modality in self.modalities if modality in obs
        }
        transf_obs = self.transform_manager(
            transf_obs, self.transf_type, device=self.device
        )
        return transf_obs

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx], allow_pickle=True)
        item = {
            "observations": self.get_transf_obs(data["state"].item()),
            "actions": data["action"],
            "next_observations": self.get_transf_obs(data["next_state"].item()),
            "rewards": data["reward"].item(),
            "terminals": data["done"].item(),
        }
        return item
