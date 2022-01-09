import hydra

from trainer import trainer


@hydra.main(config_path="conf/trainer", config_name="trainer")
def train_external(cfg):
    trainer.train(cfg)


if __name__ == '__main__':
    train_external()
