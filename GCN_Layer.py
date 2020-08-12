import tensorflow as tf
from spp.SpatialPyramidPooling import SpatialPyramidPooling

class GCN_Layer(tf.keras.layers.Layer):
    def __init__(self, num_kernels, num_neurons, qaoa_depth):
        super(GCN_Layer, self).__init__()
        self.num_neurons = num_neurons
        self.num_kernels = num_kernels

        # inputs -> shared layer -> produce hidden state, param prediction, and expectation value
        self.conv_layer = tf.keras.layers.Conv2D(filters = self.num_kernels, 
                                                kernel_size = 2, 
                                                name="conv_layer", 
                                                activation = tf.keras.activations.elu,
                                                data_format = 'channels_last')
        self.flatten = tf.keras.layers.Flatten(data_format=None)
        self.spp_layer = SpatialPyramidPooling([1, 2], dim_ordering='channels_last')
        self.dense_layer = tf.keras.layers.Dense(self.num_neurons , name="encoder_layer", activation = tf.keras.activations.elu)

    def call(self,inputs):
        conv_layer_out = self.conv_layer(inputs)
        spp_layer_out = self.spp_layer(conv_layer_out)
        dense_layer_out = self.dense_layer(spp_layer_out)
        return dense_layer_out


