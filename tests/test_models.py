import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from pndapetzim.models import CategoricalFeatureDescriptor
from pndapetzim.models import build_large_model
from pndapetzim.models import build_small_model


def test_small_model_shape():
    seq_len = 10
    batch_size = 32

    action_mask = tf.random.uniform(
        shape=(batch_size, seq_len), dtype=tf.float32
    )
    amount_paid = tf.random.uniform(
        shape=(batch_size, seq_len), dtype=tf.float32
    )
    order_date = tf.random.uniform(
        shape=(batch_size, seq_len), dtype=tf.float32
    )

    model = build_small_model(seq_len, 10)

    output = model(
        {
            'action_mask': action_mask,
            'amount_paid': amount_paid,
            'order_date': order_date,
        },
    )

    expected_shape = (batch_size, 1)

    assert output.shape == expected_shape


def test_small_model_fit():
    seq_len = 10
    batch_size = 32
    train_size = 320

    action_mask = Dataset.from_tensor_slices(
        tf.random.uniform(shape=(train_size, seq_len), dtype=tf.float32)
    )
    amount_paid = Dataset.from_tensor_slices(
        tf.random.uniform(shape=(train_size, seq_len), dtype=tf.float32)
    )
    order_date = Dataset.from_tensor_slices(
        tf.random.uniform(shape=(train_size, seq_len), dtype=tf.float32)
    )
    label = Dataset.from_tensor_slices(
        tf.random.uniform(shape=(train_size, 1), maxval=2, dtype=tf.int32)
    )
    weight = Dataset.from_tensor_slices(
        tf.random.uniform(shape=(train_size,), dtype=tf.float32)
    )

    input = action_mask, amount_paid, order_date
    ds = Dataset.zip((input, label, weight))

    def make_dict(input, label, weight):
        action_mask, amount_paid, order_date = input
        return (
            {
                'action_mask': action_mask,
                'amount_paid': amount_paid,
                'order_date': order_date,
            },
            label,
            weight,
        )

    ds = ds.map(make_dict)
    ds = ds.batch(batch_size)

    model = build_small_model(seq_len, 10)

    loss = BinaryCrossentropy()
    optimiser = Adam()

    model.compile(loss=loss, optimizer=optimiser)

    model.fit(ds, epochs=1)


def test_big_model_shape():
    seq_len = 10
    batch_size = 32

    action_mask = tf.random.uniform(
        shape=(batch_size, seq_len), dtype=tf.float32
    )
    order_date = tf.random.uniform(
        shape=(batch_size, seq_len), dtype=tf.float32
    )
    order_hour_cos = tf.random.uniform(
        shape=(batch_size, seq_len), dtype=tf.float32
    )
    order_hour_sin = tf.random.uniform(
        shape=(batch_size, seq_len), dtype=tf.float32
    )
    is_failed = tf.random.uniform(shape=(batch_size, seq_len), dtype=tf.float32)
    voucher_amount = tf.random.uniform(
        shape=(batch_size, seq_len), dtype=tf.float32
    )
    delivery_fee = tf.random.uniform(
        shape=(batch_size, seq_len), dtype=tf.float32
    )
    amount_paid = tf.random.uniform(
        shape=(batch_size, seq_len), dtype=tf.float32
    )

    # Categorigal features.
    restaurant_id = tf.random.uniform(
        shape=(batch_size, seq_len), maxval=10, dtype=tf.int32
    )
    city_id = tf.random.uniform(
        shape=(batch_size, seq_len), maxval=10, dtype=tf.int32
    )
    payment_id = tf.random.uniform(
        shape=(batch_size, seq_len), maxval=10, dtype=tf.int32
    )
    platform_id = tf.random.uniform(
        shape=(batch_size, seq_len), maxval=10, dtype=tf.int32
    )
    transmission_id = tf.random.uniform(
        shape=(batch_size, seq_len), maxval=10, dtype=tf.int32
    )

    cat_features = {
        'restaurant_id': CategoricalFeatureDescriptor(
            vocab_size=15, embedding_size=3
        ),
        'city_id': CategoricalFeatureDescriptor(
            vocab_size=15, embedding_size=3
        ),
        'payment_id': CategoricalFeatureDescriptor(
            vocab_size=15, embedding_size=3
        ),
        'platform_id': CategoricalFeatureDescriptor(
            vocab_size=15, embedding_size=3
        ),
        'transmission_id': CategoricalFeatureDescriptor(
            vocab_size=15, embedding_size=3
        ),
    }

    model = build_large_model(seq_len, cat_features)

    output = model(
        {
            'action_mask': action_mask,
            'order_date': order_date,
            'order_hour_cos': order_hour_cos,
            'order_hour_sin': order_hour_sin,
            'is_failed': is_failed,
            'voucher_amount': voucher_amount,
            'delivery_fee': delivery_fee,
            'amount_paid': amount_paid,
            'restaurant_id': restaurant_id,
            'city_id': city_id,
            'payment_id': payment_id,
            'platform_id': platform_id,
            'transmission_id': transmission_id,
        },
    )

    expected_shape = (batch_size, 1)

    assert output.shape == expected_shape
