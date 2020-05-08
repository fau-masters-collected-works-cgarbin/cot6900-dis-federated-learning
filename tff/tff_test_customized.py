#! ./env/bin/python
'''Tensorflow Federated Learning - image classification

Code from the tutorial at https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification, # noqa
adapted to run as a stand-alone Python program.

This is the customized version. It creates a customized TFF model.

Requires Python 3
'''
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import shutil
import tff_common

# Ignore TensorFlow v2 deprecation warnings to clean up the output
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.compat.v1.enable_v2_behavior()

# Help makes runs repeatable
np.random.seed(0)

tff_common.test()

# Drives a training round
# What to train: weights, bias
# What to update during training: loss, accuracy, number of examples
MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')


def create_mnist_variables():
    return MnistVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
            name='weights',
            trainable=True),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(10)),
            name='bias',
            trainable=True),
        num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
        loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
        accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


def mnist_forward_pass(variables, batch):
    print('Forward pass')
    y = tf.nn.softmax(
        tf.matmul(batch['x'], variables.weights) + variables.bias)
    predictions = tf.cast(tf.argmax(y, 1), tf.int32)

    flat_labels = tf.reshape(batch['y'], [-1])
    loss = -tf.reduce_mean(
        tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, flat_labels), tf.float32))

    num_examples = tf.cast(tf.size(batch['y']), tf.float32)

    variables.num_examples.assign_add(num_examples)
    variables.loss_sum.assign_add(loss * num_examples)
    variables.accuracy_sum.assign_add(accuracy * num_examples)

    return loss, predictions


def get_local_mnist_metrics(variables):
    return collections.OrderedDict(
        num_examples=variables.num_examples,
        loss=variables.loss_sum / variables.num_examples,
        accuracy=variables.accuracy_sum / variables.num_examples)


@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
    metrics = collections.OrderedDict(
        num_examples=tff.federated_sum(metrics.num_examples),
        loss=tff.federated_mean(metrics.loss, metrics.num_examples),
        accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))
    return metrics


class MnistModel(tff.learning.Model):
    '''Customized TFF model.

    We could have used the functions that are wrapped in this class directly,
    but creating the model allows for more customization
    '''

    def __init__(self):
        self._variables = create_mnist_variables()

    @property
    def trainable_variables(self):
        return [self._variables.weights, self._variables.bias]

    @property
    def non_trainable_variables(self):
        return []

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    @property
    def input_spec(self):
        return collections.OrderedDict(
            x=tf.TensorSpec([None, 784], tf.float32),
            y=tf.TensorSpec([None, 1], tf.int32))

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = mnist_forward_pass(self._variables, batch)
        num_exmaples = tf.shape(batch['x'])[0]
        return tff.learning.BatchOutput(
            loss=loss, predictions=predictions, num_examples=num_exmaples)

    @tf.function
    def report_local_outputs(self):
        return get_local_mnist_metrics(self._variables)

    @property
    def federated_output_computation(self):
        return aggregate_mnist_metrics_across_clients


iterative_process = tff.learning.build_federated_averaging_process(
    MnistModel,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02))
state = iterative_process.initialize()

NUM_CLIENTS = 10
train_data, test_data = tff_common.training_data(NUM_CLIENTS)


# Collect TensorBoard data
# Start TensorBoard with
#     tensorboard --logdir /tmp/logs/scalars/ --port=0
logdir = './logs/scalars/training/'
# Delete results from a previous run
shutil.rmtree(logdir, ignore_errors=True)
summary_writer = tf.summary.create_file_writer(logdir)

NUM_ROUNDS = 10
with summary_writer.as_default():
    for round_num in range(NUM_ROUNDS):
        # Here we would pick differet clients for each round in real life
        state, metrics = iterative_process.next(state, train_data)
        print('round {:2d}, metrics={}'.format(round_num+1, metrics))
        for name, value in metrics._asdict().items():
            tf.summary.scalar(name, value, step=round_num)

evaluation = tff.learning.build_federated_evaluation(MnistModel)

train_metrics = evaluation(state.model, train_data)
print('Training metrics: {}'.format(train_metrics))

test_metrics = evaluation(state.model, test_data)
print('Test metrics: {}'.format(test_metrics))
