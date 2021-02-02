from typing import Dict
import tensorflow as tf
from pydantic import BaseModel
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class CategoricalFeatureDescriptor(BaseModel):
    vocab_size: int
    embedding_size: int


def build_small_model(seq_len, hidden_layer_dim=3):
    action_mask = Input(shape=(seq_len,), dtype=tf.float32, name='action_mask')
    amount_paid = Input(shape=(seq_len,), dtype=tf.float32, name='amount_paid')
    order_date = Input(shape=(seq_len,), dtype=tf.float32, name='order_date')

    na = tf.newaxis
    y = tf.concat(
        [action_mask[:, :, na], amount_paid[:, :, na], order_date[:, :, na]],
        axis=-1,
    )

    y = Dense(hidden_layer_dim, activation='tanh')(y)
    y = Flatten()(y)

    y = Dense(10, activation='tanh')(y)

    classifier = Dense(1, activation='sigmoid')(y)

    inputs = [action_mask, amount_paid, order_date]

    outputs = classifier

    return Model(inputs=inputs, outputs=outputs)


def build_large_model(
    seq_len: int, cat_features: Dict[str, CategoricalFeatureDescriptor]
):
    na = tf.newaxis

    action_mask = Input(shape=(seq_len,), dtype=tf.float32, name='action_mask')
    order_date = Input(shape=(seq_len,), dtype=tf.float32, name='order_date')
    order_hour_cos = Input(shape=(seq_len,), dtype=tf.float32, name='order_hour_cos')
    order_hour_sin = Input(shape=(seq_len,), dtype=tf.float32, name='order_hour_sin')
    is_failed = Input(shape=(seq_len,), dtype=tf.float32, name='is_failed')
    voucher_amount = Input(shape=(seq_len,), dtype=tf.float32, name='voucher_amount')
    delivery_fee = Input(shape=(seq_len,), dtype=tf.float32, name='delivery_fee')
    amount_paid = Input(shape=(seq_len,), dtype=tf.float32, name='amount_paid')

    y = tf.concat(
        [
            action_mask[:, :, na],
            order_date[:, :, na],
            order_hour_cos[:, :, na],
            order_hour_sin[:, :, na],
            is_failed[:, :, na],
            voucher_amount[:, :, na],
            delivery_fee[:, :, na],
            amount_paid[:, :, na],
        ],
        axis=-1,
    )

    y = Flatten()(y)
    y = Dense(100)(y)

    classifier = Dense(1, activation='sigmoid')(y)

    inputs = [
        action_mask,
        order_date,
        order_hour_cos,
        order_hour_sin,
        is_failed,
        voucher_amount,
        delivery_fee,
        amount_paid,
    ]

    outputs = classifier

    return Model(inputs=inputs, outputs=outputs)
