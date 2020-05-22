# Created by kwanhoon on 15/05/2020

import neural_structured_learning as nsl

import tensorflow as tf

### Experiment dataset
TRAIN_DATA_PATH = './data/testdataset.tfr'
TEST_DATA_PATH = './data/testdataset_test.tfr'

### Constants used to identify neighbor features in the input.
NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'

class HParams(object):
  """Hyperparameters used for training."""
  def __init__(self):
    ### dataset parameters
    self.num_classes = 3
    self.max_seq_length = 7
    ### neural graph learning parameters
    self.distance_type = nsl.configs.DistanceType.L2
    self.graph_regularization_multiplier = 0.1
    self.num_neighbors = 1
    ### model architecture
    self.num_fc_units = [50, 50]
    ### training parameters
    self.train_epochs = 100
    self.batch_size = 1
    self.dropout_rate = 0.5
    ### eval parameters
    self.eval_steps = None  # All instances in the test set are evaluated.

HPARAMS = HParams()


def parse_example(example_proto):
  """Extracts relevant fields from the `example_proto`.

  Args:
    example_proto: An instance of `tf.train.Example`.

  Returns:
    A pair whose first value is a dictionary containing relevant features
    and whose second value contains the ground truth label.
  """
  # The 'words' feature is a multi-hot, bag-of-words representation of the
  # original raw text. A default value is required for examples that don't
  # have the feature.
  feature_spec = {
      # 'words':
      #     tf.io.FixedLenFeature([HPARAMS.max_seq_length],
      #                           tf.float32,
      #                           default_value=tf.constant(
      #                               0,
      #                               dtype=tf.float32,
      #                               shape=[HPARAMS.max_seq_length])),
      'x':
          tf.io.FixedLenFeature((), tf.float32, default_value=-1),
      'y':
          tf.io.FixedLenFeature((), tf.float32, default_value=-1),
      'z':
          tf.io.FixedLenFeature((), tf.float32, default_value=-1),
      'a':
          tf.io.FixedLenFeature((), tf.float32, default_value=-1),
      'ax':
          tf.io.FixedLenFeature((), tf.float32, default_value=-1),
      'ax2':
          tf.io.FixedLenFeature((), tf.float32, default_value=-1),
      'v':
          tf.io.FixedLenFeature((), tf.float32, default_value=-1),
      'label':
          tf.io.FixedLenFeature((), tf.int64, default_value=-1),
  }
  print(feature_spec)
  # We also extract corresponding neighbor features in a similar manner to
  # the features above.
  for i in range(HPARAMS.num_neighbors):
    nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'a')
    print('nbr_feature_key:', nbr_feature_key )
    nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i, NBR_WEIGHT_SUFFIX)
    print('nbr_weight_key:', nbr_weight_key)
    feature_spec[nbr_feature_key] =tf.io.FixedLenFeature((), tf.float32, default_value=-1)
    print(feature_spec[nbr_feature_key])

    # We assign a default value of 0.0 for the neighbor weight so that
    # graph regularization is done on samples based on their exact number
    # of neighbors. In other words, non-existent neighbors are discounted.

    feature_spec[nbr_weight_key] = tf.io.FixedLenFeature(
        [1], tf.float32, default_value=tf.constant([0.0]))

  features = tf.io.parse_single_example(example_proto, feature_spec)
  label = features.pop('label')
  return features, label


def make_dataset(file_path, training=False):
  """Creates a `tf.data.TFRecordDataset`.

  Args:
    file_path: Name of the file in the `.tfrecord` format containing
      `tf.train.Example` objects.
    training: Boolean indicating if we are in training mode.

  Returns:
    An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
    objects.
  """
  dataset = tf.data.TFRecordDataset([file_path])
  print("dataset:", dataset)
  if training:
    dataset = dataset.shuffle(10000)
  dataset = dataset.map(parse_example)
  dataset = dataset.batch(HPARAMS.batch_size)
  return dataset


train_dataset = make_dataset(TRAIN_DATA_PATH, training=True)
test_dataset = make_dataset(TEST_DATA_PATH)

#
# for feature_batch, label_batch in train_dataset.take(1):
#   print('Feature list:', list(feature_batch.keys()))
#   print('Batch of inputs:', feature_batch['words'])
#   nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, 0, 'words')
#   nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, 0, NBR_WEIGHT_SUFFIX)
#   print('Batch of neighbor inputs:', feature_batch[nbr_feature_key])
#   print('Batch of neighbor weights:',
#         tf.reshape(feature_batch[nbr_weight_key], [-1]))
#   print('Batch of labels:', label_batch)
#
# for feature_batch, label_batch in test_dataset.take(1):
#   print('Feature list:', list(feature_batch.keys()))
#   print('Batch of inputs:', feature_batch['words'])
#   nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, 0, 'words')
#   nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, 0, NBR_WEIGHT_SUFFIX)
#   print('Batch of neighbor inputs:', feature_batch[nbr_feature_key])
#   print('Batch of neighbor weights:',
#         tf.reshape(feature_batch[nbr_weight_key], [-1]))
#   print('Batch of labels:', label_batch)


def make_mlp_sequential_model(hparams):
  """Creates a sequential multi-layer perceptron model."""
  model = tf.keras.Sequential()
  model.add(
      tf.keras.layers.InputLayer(
          input_shape=(hparams.max_seq_length,), )) #name='words'
  # Input is already one-hot encoded in the integer format. We cast it to
  # floating point format here.
  model.add(
      tf.keras.layers.Lambda(lambda x: tf.keras.backend.cast(x, tf.float32)))
  for num_units in hparams.num_fc_units:
    model.add(tf.keras.layers.Dense(num_units, activation='relu'))
    # For sequential models, by default, Keras ensures that the 'dropout' layer
    # is invoked only during training.
    model.add(tf.keras.layers.Dropout(hparams.dropout_rate))
  model.add(tf.keras.layers.Dense(hparams.num_classes, activation='softmax'))
  return model

def make_mlp_functional_model(hparams):
  """Creates a functional API-based multi-layer perceptron model."""
  inputs = tf.keras.Input(
      shape=(1,), dtype='int64', name='a') # name='words'

  # Input is already one-hot encoded in the integer format. We cast it to
  # floating point format here.
  cur_layer = tf.keras.layers.Lambda(
      lambda x: tf.keras.backend.cast(x, tf.float32))(
          inputs)

  for num_units in hparams.num_fc_units:
    cur_layer = tf.keras.layers.Dense(num_units, activation='relu')(cur_layer)
    # For functional models, by default, Keras ensures that the 'dropout' layer
    # is invoked only during training.
    cur_layer = tf.keras.layers.Dropout(hparams.dropout_rate)(cur_layer)

  outputs = tf.keras.layers.Dense(
      hparams.num_classes, activation='softmax')(
          cur_layer)

  model = tf.keras.Model(inputs, outputs=outputs)
  return model

def make_mlp_subclass_model(hparams):
  """Creates a multi-layer perceptron subclass model in Keras."""

  class MLP(tf.keras.Model):
    """Subclass model defining a multi-layer perceptron."""

    def __init__(self):
      super(MLP, self).__init__()
      # Input is already one-hot encoded in the integer format. We create a
      # layer to cast it to floating point format here.
      self.cast_to_float_layer = tf.keras.layers.Lambda(
          lambda x: tf.keras.backend.cast(x, tf.float32))
      self.dense_layers = [
          tf.keras.layers.Dense(num_units, activation='relu')
          for num_units in hparams.num_fc_units
      ]
      self.dropout_layer = tf.keras.layers.Dropout(hparams.dropout_rate)
      self.output_layer = tf.keras.layers.Dense(
          hparams.num_classes, activation='softmax')

    def call(self, inputs, training=False):
      cur_layer = self.cast_to_float_layer(inputs['words'])
      for dense_layer in self.dense_layers:
        cur_layer = dense_layer(cur_layer)
        cur_layer = self.dropout_layer(cur_layer, training=training)

      outputs = self.output_layer(cur_layer)

      return outputs

  return MLP()

base_model_tag, base_model = 'FUNCTIONAL', make_mlp_functional_model(HPARAMS)
base_model.summary()
base_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
base_model.fit(train_dataset, epochs=HPARAMS.train_epochs, verbose=1)

# Helper function to print evaluation metrics.
def print_metrics(model_desc, eval_metrics):
  """Prints evaluation metrics.

  Args:
    model_desc: A description of the model.
    eval_metrics: A dictionary mapping metric names to corresponding values. It
      must contain the loss and accuracy metrics.
  """
  print('\n')
  print('Eval accuracy for ', model_desc, ': ', eval_metrics['accuracy'])
  print('Eval loss for ', model_desc, ': ', eval_metrics['loss'])
  if 'graph_loss' in eval_metrics:
    print('Eval graph loss for ', model_desc, ': ', eval_metrics['graph_loss'])

eval_results = dict(
    zip(base_model.metrics_names,
        base_model.evaluate(test_dataset, steps=HPARAMS.eval_steps)))
print_metrics('Base MLP model', eval_results)

# Build a new base MLP model.
base_reg_model_tag, base_reg_model = 'FUNCTIONAL', make_mlp_functional_model(
    HPARAMS)

graph_reg_config = nsl.configs.make_graph_reg_config(
    max_neighbors=HPARAMS.num_neighbors,
    multiplier=HPARAMS.graph_regularization_multiplier,
    distance_type=HPARAMS.distance_type,
    sum_over_axis=-1)
graph_reg_model = nsl.keras.GraphRegularization(base_reg_model,
                                                graph_reg_config)
graph_reg_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
graph_reg_model.fit(train_dataset, epochs=HPARAMS.train_epochs, verbose=1)

eval_results = dict(
    zip(graph_reg_model.metrics_names,
        graph_reg_model.evaluate(test_dataset, steps=HPARAMS.eval_steps)))
print_metrics('MLP + graph regularization', eval_results)