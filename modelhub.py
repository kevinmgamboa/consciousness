"""
This file contain models implemented for the project
----------------------------

"""

# -----------------------------------------------------------------------------
#                           Libraries Needed
# -----------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from helpers_and_functions import config


class simple_cnn():
    def __init__(self, param):
        # Initializing model
        self.model = None
        # parameter
        self.parameters = param
        # # Building model structure
        # self.structure()
        # # Compiling model
        # self.compile()

    def build_model_structure(self, in_shape):
        # Adds batch size = 1
        in_shape = (1,) + in_shape
        flat_size = 10
        num_filters = 5
        kernel_size = 3
        out_size = 1

        self.model = tf.keras.Sequential([
            layers.Conv2D(num_filters, kernel_size, padding='same', activation='relu',
                          kernel_initializer='he_normal', input_shape=in_shape),
            layers.Flatten(),
            layers.Dense(flat_size, activation='relu'),
            layers.Dense(out_size)
        ])

    def compile(self):
        self.model.compile(optimizer=keras.optimizers.Adam(self.parameters['lr']),
                           loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=keras.metrics.BinaryAccuracy(name='accuracy'))

class simple_cnn_2():
    def __init__(self, param):
        # Initializing model
        self.model = None
        self.feature_extractor = None
        # parameter
        self.parameters = param
        # # Building model structure
        # self.structure()
        # # Compiling model
        # self.compile()

    def build_model_structure(self, in_shape):
        # Adds batch size = 1
        #in_shape = (1,) + in_shape
        flat_size = 10
        num_filters = 5
        kernel_size = 3
        out_size = 1

        model = tf.keras.Sequential([
            layers.Conv2D(num_filters, kernel_size, padding='same', activation='relu',
                          kernel_initializer='he_normal', input_shape=in_shape),
            layers.Flatten(),
            layers.Dense(flat_size, activation='relu', name='feature_extraction'),
            layers.Dense(out_size, name='output')
        ])

        self.model = keras.Model(
            inputs=model.inputs,
            #outputs=[layer.output for layer in model.layers[-2:]]  # extracts the last two layers of the model
            outputs=model.get_layer(name="output").output,
        )

    def compile(self):
        self.model.compile(optimizer=keras.optimizers.Adam(self.parameters['lr']),
                           loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=keras.metrics.BinaryAccuracy(name='accuracy'))

class simple_cnn_2_1():
    def __init__(self, param):
        # Initializing model
        self.model = None
        self.feature_extractor = None
        # parameter
        self.parameters = param
        # # Building model structure
        # self.structure()
        # # Compiling model
        # self.compile()

    def build_model_structure(self, in_shape):
        # Adds batch size = 1
        in_shape = (1,) + in_shape
        flat_size = 10
        num_filters = 5
        kernel_size = 3
        out_size = 1

        model = tf.keras.Sequential([
            layers.Conv2D(num_filters, kernel_size, padding='same', activation='relu',
                          kernel_initializer='he_normal', input_shape=in_shape),
            layers.Flatten(),
            layers.Dense(flat_size, activation='relu', name='feature_extraction'),
            layers.Dense(out_size, name='output')
        ])

        self.model = keras.Model(
            inputs=model.inputs,
            outputs=[layer.output for layer in model.layers[-2:]]  # extracts the last two layers of the model
            #outputs=model.get_layer(name="feature_extraction").output,
        )

    def compile(self):
        self.model.compile(optimizer=keras.optimizers.Adam(self.parameters['lr']),
                           loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=keras.metrics.BinaryAccuracy(name='accuracy'))
# -----------------------------------------------------------------------------
#                           Multi-Branch Model
# -----------------------------------------------------------------------------
class multi_output_feature_model:
    def __init__(self, param, input_shape):
        # initializing model object
        self.model = None
        # initializing parameters
        self.parameters = param
        # builds the model
        self.build_model((1,)+input_shape)  # adds extra dimension to in_shape


    def spectrogram_input_branch(self, x_in):
        x = layers.Conv2D(self.parameters['num_filters'], self.parameters['kernel_size'],
                          padding='same', activation='relu', name='spec_conv_1')(x_in)
        x = layers.Flatten(name='spec_flatten')(x)
        x = layers.Dense(self.parameters['dense_units'], activation='relu', name='spec_dense_1')(x)
        x = layers.Dense(self.parameters['out_size'], name='spec_dense_2')(x)

        return x


    def hilbert_transform_branch(self, x_in):
        x = layers.Conv2D(self.parameters['num_filters'], self.parameters['kernel_size'],
                          padding='same', activation='relu', name='hilbert_conv_1')(x_in)
        x = layers.Flatten(name='hilbert_flatten')(x)
        x = layers.Dense(self.parameters['dense_units'], activation='relu', name='hilbert_dense_1')(x)
        x = layers.Dense(self.parameters['out_size'], name='hilbert_dense_2')(x)

        return x

    def build_model(self, input_shape):
        # creates the model input
        x_in = keras.Input(input_shape)
        # builds the model
        self.model = keras.models.Model(inputs=x_in,
                                        outputs=[self.spectrogram_input_branch(x_in), self.hilbert_transform_branch(x_in)],
                                        name='multi_input_feature_model')

# %%
# -----------------------------------------------------------------------------
#          Multi-Branch Model (https://keras.io/guides/functional_api/)
# -----------------------------------------------------------------------------
class multi_input_feature_model:
    def __init__(self, param, input_1, input_2):
        # initializing model object
        self.model = None
        # initializing parameters
        self.parameters = param
        # builds the model
        self.build_model_structure((1,) + input_1, (1,) + input_2)  # adds extra dimension to in_shape

    def spectrogram_input_branch(self, x_in):
        # creating model body structure
        x = layers.Conv2D(self.parameters['num_filters'], self.parameters['kernel_size'],
                          padding='same', activation='relu')(x_in)
        x = layers.Flatten()(x)
        x = layers.Dense(self.parameters['dense_units'], activation='relu')(x)
        x = layers.Dense(self.parameters['out_size'])(x)
        # defining model output
        y = keras.models.Model(inputs=x_in, outputs=x)

        return y

    def hilbert_transform_branch(self, x_in):
        # creating model body structure
        x = layers.Conv2D(self.parameters['num_filters'], self.parameters['kernel_size'],
                          padding='same', activation='relu')(x_in)
        x = layers.Flatten()(x)
        x = layers.Dense(self.parameters['dense_units'], activation='relu')(x)
        x = layers.Dense(self.parameters['out_size'])(x)
        # defines the output
        y = keras.models.Model(inputs=x_in, outputs=x)

        return y

    def build_model_structure(self, input_1, input_2):
        # creates input for spectrogram and for hilbert transform
        spec_input, hilb_input = keras.Input(input_1, name='spectrogram_input'), keras.Input(input_2,
                                                                                             name='hilbert_input')
        # creates spectrogram and hilbert transform branches
        spec_b, hilb_b = self.spectrogram_input_branch(spec_input), self.spectrogram_input_branch(hilb_input)
        # combining branches outputs via concatenation
        comb_out = layers.concatenate([spec_b.output, hilb_b.output])
        # adding final extra layers
        conscious_pred = layers.Dense(self.parameters['out_size'], name='final_out')(comb_out)
        # builds the model
        self.model = keras.models.Model(inputs=[spec_b.input, hilb_b.input],
                                        outputs=conscious_pred,
                                        name='multi_input_feature_model')

    def compile(self):
        self.model.compile(optimizer=keras.optimizers.Adam(self.parameters['lr']),
                           loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=config.metrics)