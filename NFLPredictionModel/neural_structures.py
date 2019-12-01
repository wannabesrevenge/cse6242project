from typing import List

import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras import losses
from keras.optimizers import Adadelta


def _get_activation(layer: int, activations: List[str]):
    if activations is None or len(activations) == 0:
        return 'relu'
    elif layer >= len(activations):
        return activations[-1]
    else:
        return activations[layer]


class NeuralNetwork:
    """
    Utility Class That Simplifies Creation And Training Of Networks
    """

    def __init__(self, keras_network):
        self.network: Sequential = keras_network

    def train(self, training_data, validation_data, training_details=None) -> int:
        # Train With Milestones
        if training_details is None:
            training_details = {"batch-size": 100}

        # patient early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
        # fit model
        history = self.network.fit(training_data[0], training_data[1], validation_data=(validation_data[0], validation_data[1]), epochs=1000, verbose=0, callbacks=[es], batch_size=16, shuffle=True)

        return history.epoch[-1]

    def error(self, X, y):
        out = self.network.predict(X)
        samples = len(X)
        if len(y.shape) < 2:
            y = np.reshape(y, (samples, 1))

        features = y.shape[1]

    def predict(self, X):
        return self.network.predict(X)

    def save(self, path: str):
        pass

    @staticmethod
    def load(path: str):
        # Open and load stuff...
        pass

    @staticmethod
    def create_network(features: int, outputs: int, initial: int, depth: int, decay: float = .5, activations: List[str] = None, final_layer: str = "sigmoid", loss="binary_crossentropy",dropout=.2):

        network = Sequential()
        network.add(Dense(initial, input_shape=(features,), activation=_get_activation(0, activations)))
        network.add(Dropout(dropout))
        current_size = initial
        for d in range(1, depth):
            current_size = current_size * decay
            network.add(Dense(int(current_size), activation=_get_activation(d, activations)))

        # Output Layer
        network.add(Dense(outputs, activation=final_layer))
        network.compile(loss=loss, optimizer=Adadelta())
        # print("Parameters: " + str(network.count_params()))
        return NeuralNetwork(network)
