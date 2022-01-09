from keras import Sequential
from keras.applications.resnet import ResNet50
from keras.layers import GlobalMaxPooling2D, Dense
from keras.models import load_model


class ResNetModel:
    def __init__(self, data_shape, init_weights='imagenet',
                 loss='categorical_crossentropy', optimizer='adam'):
        self.num_classes = data_shape.num_classes
        self.input_size = data_shape.image_size
        self.init_weights = init_weights
        self.loss = loss
        self.optimizer = optimizer
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(
            ResNet50(include_top=False, weights='imagenet', input_shape=self.input_size))
        self.model.add(GlobalMaxPooling2D())
        self.model.add(Dense(self.num_classes, activation='softmax', use_bias=True))
        self.model.summary()
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['acc'],
            run_eagerly=False
        )

    def get_model(self):
        return self.model

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)
        return self.model
