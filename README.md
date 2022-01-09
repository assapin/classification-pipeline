# Classification pipeline with Keras, Hydra and Clear.ml

This multi-class pipeline demo.
1. It uses [Hydra](https://hydra.cc/) to manage the configuration of model, data loading and training configuration.
2. It uses [Clear.ml](https://clear.ml/docs/latest/) to track experiments, models, datasets, and pipelines

The configuration included trains a ResNet (pretrained on imagen et) on the Microsoft [Pets](https://www.microsoft.com/en-us/download/details.aspx?id=54765) dataset

## Setup
### Prerequisites

1. Python 3.7+ 
   
2. virtualenv / conda (optional)

### Dependencies
```
pip install -r requirements.txt
```

### Clear.ml
1. Create a free account in Clear.ml website
2. Follow the instructions [here:](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps)

### Downloading the full Pets dataset (Optional)
This repo contains a sample of the Microsoft dataset, divided into train and test.
If you would like to download the full dataset, you can find it [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765) dataset


## Codebase structure
```
.
├── data
│   ├── test
│   │   ├── cat
│   │   ├── dog
│   └── train
│       ├── cat
│       ├── dog
└── src
    ├── conf
    │   ├── data_pipeline
    │   └── trainer
    │       ├── dataloader
    │       │   └── dataset
    │       ├── model
    │       │   └── optimizer
    │       └── tracker
    ├── data_pipeline
    ├── evaluator
    └── trainer
        ├── dataloader
        │   └── dataset
        └── models


```

### conf folder
Each package/module has an equivalent config "package" with potentially multiple settings.

For example, the model config folder has an optimizer sub-folder.
This sub folder contain configurations that specify how to instantiate and configure different optimizers.
By default, the model uses the 'adam.yaml' configuration.

### data_pipeline module
Hydra driven ETL code that extracts, transforms and loads data.
Configured to work with Clear.ml datasets

### trainer module
In charge of the training loop, publishing checkpoints, and metrics to Tensorboard
```trainer.py``` contains the main training entrypoint. 


#### dataloader package
In charge of the loading process of data - from augmentation to batching.
Points to a ```dataset``` object that contains details about the actual images

##### dataset package
Contains the configuration for actually loading images from file system, including metadata on the image format

#### model package 
In charge of the model architecture, optimizers, learning rates etc.
Provided a ResNet model implementation

## Using Hydra to drive configuration
Using Hydra, you can override any configuration purely from the command line.
Each run writes 
1. output folder with the final configuration used post overrides
2. Log files to the file system.

for example, to override an object, such as the dataset we use for training - trigger the trainer as follows:
```python src/trainer/trainer.py dataloader/dataset=pets```

to override a primitive value, use the dot notation:
```python src/trainer/trainer.py  dataloader.augmentations.horizontal_flip=False```

### Sweeps
Using a simple syntax, Hydra allows you to launch multiple instances of your program, with differnt configuration
variants.

Using a launcher, you can even specify that these different experiments run in parallel to each other.

Example:
```python train.py  dataloader.augmentations.horizontal_flip=False,True hydra/launcher=joblib -m```

This command:
1. Uses the ```joblib``` job launcher
2. Fires 2 experiemnts concurrently:
   a. horizontal_flip=False
   b. horizontal_flip=True
   
Each experiment will write its logs to a folder such as this:

```
└── 2021-09-05
    └── 18-36-15
        ├── 0
        │   └── classifier.log
        ├── 1
        │   └── classifier.log
        └── multirun.yaml
```

## Using Clear.ml to track experiments, artefacts and pipelines

### How this codebase interacts with Clear.ml
1. The ```trainer``` and the ```dataset``` modules have a flag called "track".
   By default they are set to "false".
   * If the trainer's flag is set to "true", it will create a new experiment each time you run the trainer.py script
     and log all the environment including the hydra configuraiton
   * If the dataset's flag is set to "true", it will expect to load the data from Clear.ml's Data versioning API
      and not from your local file system
   
2. The "data_pipeline.py" module uploads the (processed) dataset to Clear.ml data versioning API

3. The "masks_pipeline.py" that runs both the data_pipeline.py and the trainer.py is configured to clone tasks
   that already exist within the Clear.ml system.
   In order to run it, you need to first run the data_pipeline.py and the trainer.py with "track=true"
   
