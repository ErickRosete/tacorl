import torch
import torch.distributed as dist
from pytorch_lightning.loggers import WandbLogger


class MetricsLogger:
    """Custom Wandb Metrics Logger where only rank 0 can log metrics"""

    def __init__(self, logger: WandbLogger):
        self.logger = logger

    def log(
        self,
        info: dict,
        log_type: str = "validation",
        step: int = None,
        device: str = "cuda",
    ):
        """Use wandb log option to log the metrics,
        when it is not possible to use pl_module.log"""

        if dist.is_available() and dist.is_initialized():
            all_values = {}
            for key in info.keys():
                value = torch.tensor(info[key], dtype=torch.float, device=device)
                all_values[key] = [
                    torch.zeros_like(value, dtype=torch.float, device=device)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(all_values[key], value)

            if dist.get_rank() != 0:
                return
            log_info = {}
            for key in all_values.keys():
                log_info[key] = torch.stack(all_values[key], axis=0).mean().item()
        else:
            log_info = info

        wandb_logs = {
            "%s/%s" % (log_type, key): log_info[key] for key in log_info.keys()
        }
        self.logger.log_metrics(wandb_logs, step=step)
