import logging

import hydra
import yaml
from clearml import TaskTypes, PipelineDecorator
from omegaconf import OmegaConf

from data_pipeline.pets import etl
from trainer import trainer

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    params = yaml.safe_load(OmegaConf.to_yaml(cfg, resolve=True))
    do_pipeline(params)


@PipelineDecorator.component(execution_queue='default', return_values=['dataset_id'],task_type=TaskTypes.data_processing)
def wrap_etl(params):
    dataset_id = etl(params)
    return dataset_id


@PipelineDecorator.component(execution_queue='default', return_values=['dataset_id'], task_type=TaskTypes.training)
def wrap_trainer(params):
    return trainer.train(params)


@PipelineDecorator.pipeline(name='pets-training-pipeline', project='classification-example', version='1.0', default_queue='default' )
def do_pipeline(params):
    dataset_id = wrap_etl(params['data_pipeline'])
    print(params['trainer']['dataloader']['dataset']['dataset_id'])
    params['trainer']['dataloader']['dataset']['dataset_id'] = dataset_id
    trained = wrap_trainer(params['trainer'])
    print(trained)


if __name__ == '__main__':
    PipelineDecorator.run_locally()
    main()
