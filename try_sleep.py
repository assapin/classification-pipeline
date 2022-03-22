import argparse
from time import sleep

from clearml import PipelineDecorator, Task, TaskTypes

@PipelineDecorator.component(execution_queue='default', return_values=['message'], task_type=TaskTypes.data_processing)
def get_dateset_id():
    message = "ccd8a65770e1407394cd3648246e4d25"
    return message

@PipelineDecorator.component(execution_queue='default', return_values=['message2'], task_type=TaskTypes.data_processing)
def after(message):
    message2 = message + "returned!!"
    return message2


@PipelineDecorator.pipeline(name='try-aborting-and-restarting', project='classification-example', version='1.0', default_queue='default', )
#
def logic():
    message = get_dateset_id()
    print(message)
    from clearml import Dataset
    ds = Dataset.get(dataset_id=message, dataset_tags='released')
    if not ds or 'released' not in ds.tags:
        print("aborting ourselves")
        Task.current_task().mark_stopped()
        # we will not get here, the agent will make sure we are stopped
        sleep(60)
        # better safe than sorry
        exit(0)
    message2 = after(message)
    print(message2)


if __name__ == '__main__':
    PipelineDecorator.run_locally()
    logic()