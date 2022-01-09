from keras_preprocessing.image import ImageDataGenerator


class GeneratorDataLoader:
    def __init__(self, dataset, class_mode='categorical', batch_size=32, augmentations = {}):
        self.train_dir = dataset.train_dir
        self.val_dir= dataset.val_dir
        self.class_mode = class_mode
        self.image_size = dataset.data_shape.image_size[0:2] #ignore channels
        self.batch_size = batch_size
        self.train_aug = augmentations
        self.train_datagen = ImageDataGenerator(**augmentations)
        self.val_datagen = ImageDataGenerator()

    def train_generator(self):
        return self.train_datagen.flow_from_directory(
            directory=self.train_dir,
            target_size=self.image_size,
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode=self.class_mode)

    def val_generator(self):
        return self.val_datagen.flow_from_directory(
            directory=self.val_dir,
            target_size=self.image_size,
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode=self.class_mode)

