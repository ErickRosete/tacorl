import io
import logging
from typing import Any, Optional

import hydra
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from MulticoreTSNE import MulticoreTSNE as TSNE
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

log = logging.getLogger(__name__)


def plotly_fig2array(fig):
    """convert Plotly fig to  an array"""
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


class TSNEPlot(Callback):
    def __init__(
        self,
        perplexity,
        n_jobs,
        plot_percentage,
        opacity,
        marker_size,
        tasks: DictConfig,
    ):
        self.perplexity = perplexity
        self.n_jobs = n_jobs
        self.plot_percentage = plot_percentage
        self.opacity = opacity
        self.marker_size = marker_size
        tasks = hydra.utils.instantiate(tasks)
        self.id_to_task = tasks.id_to_task
        self.task_to_id = tasks.task_to_id
        self.sampled_plans = []
        self.labels = []

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # Only plot plans for sequences with 0 or 1 completed task
        sampled_plans, labels = [], []
        for i, completed_tasks in enumerate(outputs["completed_tasks"]):
            if len(completed_tasks) <= 1:
                sampled_plans.append(outputs["sampled_plan_pp"][i])
                if len(completed_tasks) == 1:
                    labels.append(self.task_to_id[list(completed_tasks)[0]])
                else:
                    labels.append(-1)

        if len(sampled_plans) > 0:
            self.sampled_plans.append(torch.stack(sampled_plans, dim=0))
            self.labels.append(torch.LongTensor(labels))

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if pl_module.global_step > 0:

            sampled_plans = torch.cat(self.sampled_plans, dim=0)
            labels = torch.cat(self.labels, dim=0)
            self.sampled_plans = []
            self.labels = []

            if dist.is_available() and dist.is_initialized():
                sampled_plans = pl_module.all_gather(sampled_plans)
                labels = pl_module.all_gather(labels)

                if dist.get_rank() != 0:
                    return

            x_tsne = self._get_tsne(sampled_plans)
            self._create_tsne_figure(
                labels=labels,
                x_tsne=x_tsne,
                step=pl_module.global_step,
                logger=pl_module.logger,
                name="task_consistency",
            )

    def _get_tsne(self, sampled_plans):
        x_tsne = TSNE(perplexity=self.perplexity, n_jobs=self.n_jobs).fit_transform(
            sampled_plans.view(-1, sampled_plans.shape[-1]).cpu()
        )
        return x_tsne

    def _create_tsne_figure(self, labels, x_tsne, step, logger, name):
        """
        compute t-SNE plot of embeddings os a task
        to visualize temporal consistency
        """
        labels = labels.flatten().cpu().numpy()
        assert x_tsne.shape[0] == len(
            labels
        ), f"plt X shape {x_tsne.shape[0]}, label len {len(labels)}"

        n = np.where(labels == -1)[0]
        non_task_ids = np.random.choice(
            n,
            replace=False,
            size=min(int(len(n) * self.plot_percentage), 100),
        )

        n = np.where(labels != -1)[0]
        task_ids = np.random.choice(
            n,
            replace=False,
            size=int(len(n) * self.plot_percentage),
        )
        tasks = [self.id_to_task[i] for i in labels[task_ids]]
        symbol_seq = ["circle", "square", "diamond", "cross"]

        fig = go.Figure()
        if len(non_task_ids) != 0:
            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    x=x_tsne[[non_task_ids], 0].flatten(),
                    y=x_tsne[[non_task_ids], 1].flatten(),
                    opacity=self.opacity,
                    marker={"color": "black", "size": self.marker_size},
                    showlegend=True,
                    name="no task",
                )
            )
        if len(task_ids) != 0:
            task_scatter = px.scatter(
                x=x_tsne[[task_ids], 0].flatten(),
                y=x_tsne[[task_ids], 1].flatten(),
                color=tasks,
                color_discrete_sequence=px.colors.qualitative.Alphabet,
                symbol=labels[task_ids],
                symbol_sequence=symbol_seq,
                labels={"color": "Tasks"},
            )
            for scatter in task_scatter.data:
                fig.add_trace(scatter)
        self._log_figure(fig, logger, step, name)

    @staticmethod
    def _log_figure(fig, logger, step, name):
        if isinstance(logger, WandbLogger):
            logger.experiment.log({name: fig})
        else:
            logger.experiment.add_image(name, plotly_fig2array(fig), global_step=step)
