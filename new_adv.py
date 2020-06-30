# Created by kwanhoon on 22/05/2020

import neural_structured_learning as nsl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

import pandas as pd

class HParams(object):
  """Hyperparameters used for training."""
  def __init__(self):
    ### dataset parameters
    self.num_classes = 8
    self.max_seq_length = 8
    ### neural graph learning parameters
    self.distance_type = nsl.configs.DistanceType.L2
    self.graph_regularization_multiplier = 0.1
    self.num_neighbors = 5
    ### model architecture
    self.num_fc_units = [50, 50]
    ### training parameters
    self.train_epochs = 100
    self.batch_size = 1
    self.dropout_rate = 0.5
    ### eval parameters
    self.eval_steps = None  # All instances in the test set are evaluated.

HPARAMS = HParams()
def make_mlp_functional_model(hparams):
  """Creates a functional API-based multi-layer perceptron model."""
  inputs = tf.keras.Input(
      shape=(hparams.max_seq_length,),  name='beam_features') # name='words'dtype='int64',

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

df = pd.read_csv('./data/contents.csv', index_col=0, header=None)

X = df[df.columns.values[:8]]
Y = df[9]
enc = LabelEncoder()
enc.classes_
Y = enc.fit_transform(Y.values).reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2075, random_state=0)

base_model = make_mlp_functional_model(HPARAMS)

adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)
adv_model = nsl.keras.AdversarialRegularization(base_model, adv_config=adv_config)

# Compile, train, and evaluate.
adv_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

adv_model.fit({'feature': X_train.values, 'label': Y_train}, batch_size=1, epochs=100)
adv_model.evaluate({'feature': X_test.values, 'label': Y_test})
adv_model.save_weights('./model/adv.h5')
results = adv_model.predict({'feature':X_test.values, 'label': Y_test})
#
import pandas as pd
import numpy as np
#
# results = base_model.predict(test_dataset)
pd.DataFrame(results).to_csv('./results/adv/prob.csv')
y_pred = []
y_true = Y_test.reshape(-1)

for i in range(len(results)):
    y_pred.append(np.argmax(results[i], axis=0))

labels = pd.DataFrame([y_true, y_pred]).T
labels.columns = ['y_true', 'y_pred']
labels.to_csv('./results/adv/labels.csv')