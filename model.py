import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, Activation, Flatten
from tensorflow.keras import Model


class encoder(Model):
    def __init__(self, n_dim=2, name="encoder"):
        super(encoder, self).__init__(name=name)
        self.n_dim = n_dim

        # sub layers
        self.flatten = Flatten()
        self.dense1 = Dense(1000)
        self.dense2 = Dense(1000)
        self.dense3 = Dense(self.n_dim)
        self.relu = ReLU()

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x

class decoder(Model):
    def __init__(self, name="decoder"):
        super(decoder, self).__init__(name=name)

        # sub layers
        self.dense1 = Dense(1000)
        self.dense2 = Dense(1000)
        self.dense3 = Dense(28*28)
        self.relu = ReLU()
        self.sigmoid = Activation('sigmoid')

    def call(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.sigmoid(x)
        x = tf.reshape(x, (-1, 28, 28, 1))

        return x

class translator(Model):
    def __init__(self, n_dim=2, name="translator"):
        super(translator, self).__init__(name=name)
        self.n_dim = n_dim

        # sub layers
        self.dense1 = Dense(1000)
        self.dense2 = Dense(1000)
        self.dense3 = Dense(self.n_dim)
        self.relu = ReLU()

    def call(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x

class discriminator(Model):
    def __init__(self, name="discriminator"):
        super(discriminator, self).__init__(name=name)

        # sub layers
        self.dense1 = Dense(1000)
        self.dense2 = Dense(1000)
        self.dense3 = Dense(1)
        self.relu = ReLU()

    def call(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x