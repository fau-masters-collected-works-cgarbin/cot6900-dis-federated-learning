'''Tensorflow Federated Learning - image classification

Code from the tutorial at https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification, # noqa
adapted to run as a stand-alone Python program.

These are the parts that are common to the standard and customized TFF examples.

Requires Python 3
'''
import tensorflow as tf
import tensorflow_federated as tff
import collections


def test():
    ''' Check that everything is in place.'''
    test = tff.federated_computation(lambda: 'Hello, World!')()
    assert test == b'Hello, World!'
    print(test)


def training_data(num_clients):
    '''Prepare training data for a number of clients.'''
    print('Preparing train/test data for {} clients'.format(num_clients))

    # Load the test/train dataset used in the experiments
    # EMNIST keyed by author, making it "federated", i.e. we can simulate
    # different users (authors) in a network
    train, test = tff.simulation.datasets.emnist.load_data()

    # Sample a set of clients for training
    # Simplification from the real case: we will use the same clients for all
    # training rounds. In real life we would choose a different set of clients
    # for each round. But that increases convergence time, making it
    # impractical for this exercise.
    sample_clients = train.client_ids[0:num_clients]

    return _make_federated_data(train, sample_clients), \
        _make_federated_data(test, sample_clients)


NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def _make_federated_data(client_data, client_ids):
    return [
        _preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]


def _preprocess(dataset):
    '''Prepare training data for one client (one user).

    Arguments:
        dataset -- Dataset for that client.
    '''

    def batch_format_fn(element):
        '''Flatten a batch `pixels` and return the features as an
             `OrderedDict`.'''
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)
