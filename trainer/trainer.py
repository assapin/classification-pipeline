import logging
import os
from types import SimpleNamespace

import hydra
import tensorflow as tf
from clearml import Task
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils import settings

log = logging.getLogger(__name__)


class BasicTrainer:
    def __init__(self, model, train_loader, val_loader, epochs, checkpoint_dir):
        self.model = model
        self.train_data = train_loader
        self.validation_data = val_loader
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, mode=0o777, exist_ok=True)
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'model.{epoch:02d}-{val_loss:.2f}.h5')),
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        ]

    def train(self):
        self.model.fit(self.train_data,
                       epochs=self.epochs,
                       validation_data=self.validation_data,
                       callbacks=self.callbacks
                       )
        return BasicTrainer.find_best_model(self.checkpoint_dir)

    @staticmethod
    def find_best_model(checkpoint_dir):
        import os
        file_stats = {}
        for file in os.listdir(checkpoint_dir):
            if file.endswith(".h5"):
                stat = file[:-3].split("-")[-1]
                file_stats[float(stat)] = file

        min_loss = min(file_stats.keys())
        return file_stats[min_loss]


def train(cfg):
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)
    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    loader = instantiate(cfg.dataloader)
    model = instantiate(cfg.model, cfg.dataloader.dataset.data_shape)

    trainer = BasicTrainer(model.get_model(), loader.train_generator(), loader.val_generator(),
                           cfg.trainer.epochs, cfg.trainer.checkpoint_dir)
    if cfg.track:
        task = Task.init(project_name=settings.PROJECT_NAME, task_name=cfg.name, output_uri=True)
    best_model = trainer.train()
    if cfg.track:
        task.upload_artifact("best_model", best_model)
    return best_model


