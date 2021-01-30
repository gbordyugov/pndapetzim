import tensorflow as tf

from pndapetzim.model import build_amount_date_model


def test_amount_date_model_shape():
    seq_len = 10
    batch_size = 32

    amount_paid = tf.random.uniform(shape=(batch_size, seq_len), dtype=tf.float32)
    order_date = tf.random.uniform(shape=(batch_size, seq_len), dtype=tf.float32)

    model = build_amount_date_model(seq_len, 10)

    # output = model({'amount_paid': amount_paid, 'order_date': order_date})
    output = model((amount_paid, order_date))

    expected_shape = (batch_size,)

    assert output.shape == expected_shape
