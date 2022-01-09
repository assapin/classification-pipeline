import hydra

from data_pipeline.pets import etl


@hydra.main(config_path="conf/data_pipeline", config_name="data_pipeline")
def etl_external(cfg):
    etl(cfg)


if __name__ == '__main__':
    etl_external()
