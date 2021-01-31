import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from pndapetzim.model import build_amount_date_model


def test_amount_date_model_shape():
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

    model = build_amount_date_model(seq_len, 10)

    output = model(
        {
            'action_mask': action_mask,
            'amount_paid': amount_paid,
            'order_date': order_date,
        },
    )

    expected_shape = (batch_size, 2)

    assert output.shape == expected_shape


def test_amount_date_model_fit():
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
        tf.random.uniform(shape=(train_size,), maxval=2, dtype=tf.int32)
    )

    input = action_mask, amount_paid, order_date
    ds = Dataset.zip((input, label))

    def make_dict(input, label):
        action_mask, amount_paid, order_date = input
        return (
            {
                'action_mask': action_mask,
                'amount_paid': amount_paid,
                'order_date': order_date,
            },
            label,
        )

    ds = ds.map(make_dict)
    ds = ds.batch(batch_size)


    model = build_amount_date_model(seq_len, 10)

    loss = SparseCategoricalCrossentropy()
    optimiser = Adam()

    model.compile(loss=loss, optimizer=optimiser)

    model.fit(ds, epochs=1)
