import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def build_amount_date_model(seq_len, hidden_layer_dim=10):
    action_mask = Input(shape=(seq_len,), dtype=tf.float32, name='action_mask')
    amount_paid = Input(shape=(seq_len,), dtype=tf.float32, name='amount_paid')
    order_date = Input(shape=(seq_len,), dtype=tf.float32, name='order_date')

    na = tf.newaxis
    y = tf.concat(
        [action_mask[:, :, na], amount_paid[:, :, na], order_date[:, :, na]],
        axis=-1,
    )

    y = Dense(hidden_layer_dim)(y)
    y = Flatten()(y)

    classifier = Dense(1, activation='sigmoid')(y)

    inputs = [action_mask, amount_paid, order_date]

    outputs = classifier

    return Model(inputs=inputs, outputs=outputs)
