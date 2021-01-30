import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def build_amount_date_model(seq_len, hidden_layer_dim=10):
    na = tf.newaxis

    amount_paid = Input(shape=(seq_len,), dtype=tf.float32, name='amount_paid')
    order_date = Input(shape=(seq_len,), dtype=tf.float32, name='order_date')

    concatenation = tf.concat(
        [amount_paid[:, :, na], order_date[:, :, na]], axis=-1
    )

    y = Dense(hidden_layer_dim)(concatenation)

    flat = Flatten()(y)

    classifier = Dense(2, activation='softmax')(flat)

    inputs = [amount_paid, order_date]

    outputs = classifier

    return Model(inputs=inputs, outputs=outputs)
