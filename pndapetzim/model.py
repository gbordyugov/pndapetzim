import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def build_amount_date_model(seq_len, hidden_layer_dim):
    na = tf.newaxis

    amount_paid = Input(shape=(seq_len,), dtype=tf.float32, name='amount_paid')
    order_date = Input(shape=(seq_len,), dtype=tf.float32, name='order_date')

    concatenation = tf.concat(
        [amount_paid[:, :, na], order_date[:, :, na]], axis=-1
    )

    flat = Flatten()(concatenation)

    classifier = Dense(1, activation='sigmoid')(flat)

    flattened_classifier = tf.reduce_sum(classifier, axis=1)

    inputs = [amount_paid, order_date]

    outputs = flattened_classifier

    return Model(inputs=inputs, outputs=outputs)
