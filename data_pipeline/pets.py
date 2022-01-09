import logging
import os
import shutil
from types import SimpleNamespace

import hydra
from clearml import Task

from utils import settings

log = logging.getLogger(__name__)

TASK_NAME = "pets preprocessing"


def extract(ds, target_dir):
    import os
    os.makedirs(target_dir, exist_ok=True)
    ds.get_mutable_local_copy(target_dir)
    import pathlib
    log.info(pathlib.Path(target_dir).resolve())
    return target_dir


def init(parents):
    from clearml import Dataset
    from clearml import Task

    data_task = Task.init(project_name=settings.PROJECT_NAME, task_name=TASK_NAME)
    ds = Dataset.create(dataset_name='pets_processed',
                        dataset_project=settings.PROJECT_NAME,
                        parent_datasets=parents, use_current_task=True)
    data_task.upload_artifact("dataset_id", ds.id)
    return ds


def transform(folder):
    for path, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            if name.endswith(".jpg") and hash(name) % 10 == 0:
                new_file = name + ".copy"
                shutil.copyfile(os.path.join(path, name), os.path.join(path, new_file))


def load(ds, folder):
    import pathlib
    log.info(pathlib.Path(folder).resolve())
    ds.add_files(folder)
    ds.upload()
    ds.finalize()
    return ds.id


def get_last_completed_task():
    task_filter = {
        # filter out archived Tasks
        'system_tags': ['-archived'],
        # only completed & published Tasks
        'status': ['completed', 'published'],
        # only training type Tasks
        'type': ['data_processing'],
    }
    tasks = Task.get_tasks(project_name=settings.PROJECT_NAME, task_name=TASK_NAME,
                           task_filter=task_filter)
    if tasks:
        tasks.sort(key=lambda x: x.data.completed, reverse=True)
        return [tasks[0].id]
    return None


def get_parent_ds(cfg):
    result = cfg.get("parent_ds")
    if result:
        return result
    return get_last_completed_task()


def etl(cfg):
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)
    if cfg.get("bootstrap_folder"):
        ds = init(None)
        return load(ds, cfg.bootstrap_folder)
    else:
        parent_ds = get_parent_ds(cfg)
        print(f"parent datasets:{parent_ds}")
        ds = init(parent_ds)
        folder = extract(ds, cfg.target_dir)
        transform(folder)
        return load(ds, folder)


