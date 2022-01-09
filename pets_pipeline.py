import logging

import hydra
import yaml
from clearml import PipelineController
from omegaconf import OmegaConf

from utils import settings

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    pipe = PipelineController(
        name='pipeline demo',
        project=settings.PROJECT_NAME,
        version='0.0.1',
        add_pipeline_tags=False,

    )

    pipe.set_default_execution_queue('default')

    params = yaml.safe_load(OmegaConf.to_yaml(cfg, resolve=True))
    pipe.add_step(name='data_processing', parents=[],
                  base_task_project=settings.PROJECT_NAME,
                  base_task_name='pets preprocessing',
                  base_task_id='31c08a33a7494de1905c671404880308',
                  parameter_override={
                      'configuration/OmegaConf': dict(value=params['data_pipeline'],
                                                      name='OmegaConf',
                                                      type='OmegaConf YAML')}
                  )

    print(params['trainer']['dataloader']['dataset']['dataset_id'])
    pipe.add_step(name='train', parents=['data_processing'],
                  base_task_project=settings.PROJECT_NAME,
                  base_task_name='pets training',
                  base_task_id='528879824b7842f0a615576c27a13ea8',
                  parameter_override={
                      'configuration/OmegaConf': dict(value=params['trainer'],
                                                      name='OmegaConf',
                                                      type='OmegaConf YAML'),
                      'Hydra/dataloader.dataset.dataset_id': '${data_processing.id}'
                  }
                  )

    pipe.start(queue='default')


if __name__ == '__main__':
    main()
