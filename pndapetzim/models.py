from typing import Dict
import tensorflow as tf
from pydantic import BaseModel
from tensorflow.keras.layers import Embedding
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

    # Categorical features.
    restaurant_id = Input(shape=(seq_len,), dtype=tf.int32, name='restaurant_id')
    city_id = Input(shape=(seq_len,), dtype=tf.int32, name='city_id')
    payment_id = Input(shape=(seq_len,), dtype=tf.int32, name='payment_id')
    platform_id = Input(shape=(seq_len,), dtype=tf.int32, name='platform_id')
    transmission_id = Input(shape=(seq_len,), dtype=tf.int32, name='transmission_id')

    restaurant = Embedding(
        input_dim=cat_features['restaurant_id'].vocab_size,
        output_dim=cat_features['restaurant_id'] .embedding_size,
    )(restaurant_id)

    city = Embedding(
        input_dim=cat_features['city_id'].vocab_size,
        output_dim=cat_features['city_id'] .embedding_size,
    )(city_id)

    payment = Embedding(
        input_dim=cat_features['payment_id'].vocab_size,
        output_dim=cat_features['payment_id'] .embedding_size,
    )(payment_id)

    platform = Embedding(
        input_dim=cat_features['platform_id'].vocab_size,
        output_dim=cat_features['platform_id'] .embedding_size,
    )(platform_id)

    transmission = Embedding(
        input_dim=cat_features['transmission_id'].vocab_size,
        output_dim=cat_features['transmission_id'] .embedding_size,
    )(transmission_id)

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

            # Categorical features.
            restaurant,
            city,
            payment,
            platform,
            transmission,
        ],
        axis=-1,
    )

    y = Dense(30, activation='tanh')(y)
    y = Dense(30, activation='tanh')(y)

    y = Flatten()(y)
    y = Dense(200, activation='tanh')(y)
    y = Dense(200, activation='tanh')(y)

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

        # Categorical features.
        restaurant_id,
        city_id,
        payment_id,
        platform_id,
        transmission_id,
    ]

    outputs = classifier

    return Model(inputs=inputs, outputs=outputs)
