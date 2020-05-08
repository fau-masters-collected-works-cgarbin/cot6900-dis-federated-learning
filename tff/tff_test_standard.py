#! ./env/bin/python
'''Tensorflow Federated Learning - image classification

Code from the tutorial at https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification, # noqa
adapted to run as a stand-alone Python program.

This is the 'standard' version. It uses the standard TFF APIs, without any
customization.

Requires Python 3
'''
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import shutil
import tff_common

# Ignore TensorFlow v2 deprecation warnings to clean up the output
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.compat.v1.enable_v2_behavior()

# Help makes runs repeatable
np.random.seed(0)

tff_common.test()

NUM_CLIENTS = 10
train_data, _ = tff_common.training_data(NUM_CLIENTS)


def create_keras_model():
    '''The basic model.

       It will be wrapped into a TFF model later. Keeping it as a separate
       function allows us to test the model outside of TFF'''
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


def model_fn():
    '''We _must_ create a new model here, and _not_ capture it from an external
       scope. TFF will call this within different graph contexts.'''
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

# Collect TensorBoard data
# Start TensorBoard with
#     tensorboard --logdir /tmp/logs/scalars/ --port=0
logdir = "./logs/scalars/training/"
summary_writer = tf.summary.create_file_writer(logdir)
state = iterative_process.initialize()

NUM_ROUNDS = 10
with summary_writer.as_default():
    for round_num in range(NUM_ROUNDS):
        state, metrics = iterative_process.next(state, train_data)
        print('round {:2d}, metrics={}'.format(round_num+1, metrics))
        for name, value in metrics._asdict().items():
            tf.summary.scalar(name, value, step=round_num)
