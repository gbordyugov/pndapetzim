import numpy as np
import tensorflow as tf
from pandas import DataFrame
from pandas import to_datetime

from pndapetzim.data import IntegerEncoding
from pndapetzim.data import encode_df
from pndapetzim.data import encode_int_column
from pndapetzim.data import get_dataset_from_df
from pndapetzim.data import normalise_dates
from pndapetzim.data import pad_left


def test_integer_encoding():
    values = [18, 13, 16, 18, 19]
    expected = IntegerEncoding(
        ix_to_value={1: 13, 2: 16, 3: 18, 4: 19},
        value_to_ix={13: 1, 16: 2, 18: 3, 19: 4},
    )
    got = IntegerEncoding.fromValues(values)

    assert got == expected
    assert got.vocab_size == 5


def test_encode_int_column():
    df = DataFrame(
        {
            'int_column': [99, 98, 93, 97, 94],
            'float_column': [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    expected_encoding = IntegerEncoding(
        value_to_ix={93: 1, 94: 2, 97: 3, 98: 4, 99: 5},
        ix_to_value={1: 93, 2: 94, 3: 97, 4: 98, 5: 99},
    )

    expected_df = DataFrame(
        {
            'int_column': [5, 4, 1, 3, 2],
            'float_column': [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    expected_columns = set(df.columns)

    got_df, got_encoding = encode_int_column(df, 'int_column')

    assert got_encoding == expected_encoding
    assert set(got_df.columns) == expected_columns

    for c in ['int_column', 'float_column']:
        assert all(got_df[c] == expected_df[c])


def test_encode_df():
    df = DataFrame(
        {
            'a': [13, 12, 11],
            'b': [1, 2, 3],
            'c': [23, 21, 22],
        }
    )

    columns = ['a', 'c']

    expected_df = DataFrame(
        {
            'a': [3, 2, 1],
            'b': [1, 2, 3],
            'c': [3, 1, 2],
        }
    )

    expected_encodings = {
        'a': IntegerEncoding(
            value_to_ix={11: 1, 12: 2, 13: 3},
            ix_to_value={1: 11, 2: 12, 3: 13},
        ),
        'c': IntegerEncoding(
            value_to_ix={21: 1, 22: 2, 23: 3},
            ix_to_value={1: 21, 2: 22, 3: 23},
        ),
    }

    got_df, got_encodings = encode_df(df, columns)

    assert got_encodings == expected_encodings
    assert set(got_df.columns) == set(expected_df.columns)

    for c in ['a', 'b', 'c']:
        assert all(got_df[c] == expected_df[c])


def test_pad_left():
    target_seq_len = 5

    inputs_and_ouputs = [
        ([], [0, 0, 0, 0, 0]),
        ([1], [0, 0, 0, 0, 1]),
        ([1, 2], [0, 0, 0, 1, 2]),
        ([1, 2, 3], [0, 0, 1, 2, 3]),
        ([1, 2, 3, 4], [0, 1, 2, 3, 4]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6]),
    ]

    for input, expected in inputs_and_ouputs:
        got = pad_left(np.array(input), target_seq_len)
        assert all(got == expected)


def test_get_dataset_from_df():
    seq_len = 5

    from_ts = to_datetime('2020-01-10')
    to_ts = to_datetime('2020-01-20')

    returning_weight = 3.0

    df = DataFrame(
        {
            'customer_id': [1, 1, 1, 2, 2],
            'order_date': [
                to_datetime('2020-01-10'),
                to_datetime('2020-01-15'),
                to_datetime('2020-01-20'),
                to_datetime('2020-01-10'),
                to_datetime('2020-01-20'),
            ],
            'amount_paid': [10.0, 20.0, 30.0, 40.0, 50.0],
            'is_returning_customer': [1, 1, 1, 0, 0],
        }
    )

    ds = get_dataset_from_df(df, seq_len, returning_weight, from_ts, to_ts)

    expected_first = {
        'action_mask': tf.constant(
            [
                0,
                0,
                1,
                1,
                1,
            ],
            dtype=tf.float32,
        ),
        'amount_paid': tf.constant(
            [0.0, 0.0, 10.0, 20.0, 30.0], dtype=tf.float32
        ),
        'order_date': tf.constant([0.0, 0.0, 0.0, 0.5, 1.0], dtype=tf.float32),
        'is_returning_customer': tf.constant(1, dtype=tf.int32),
        'weight': returning_weight,
    }

    expected_second = {
        'action_mask': tf.constant(
            [
                0,
                0,
                0,
                1,
                1,
            ],
            dtype=tf.float32,
        ),
        'amount_paid': tf.constant(
            [0.0, 0.0, 0.0, 40.0, 50.0], dtype=tf.float32
        ),
        'order_date': tf.constant([0.0, 0.0, 0.0, 0.0, 1.0], dtype=tf.float32),
        'is_returning_customer': tf.constant(0, dtype=tf.int32),
        'weight': 1.0,
    }

    for got, expected in zip(ds, [expected_first, expected_second]):
        input = got[0]
        label = got[1]
        weight = got[2]

        tf.debugging.assert_equal(input['action_mask'], expected['action_mask'])
        tf.debugging.assert_equal(input['amount_paid'], expected['amount_paid'])
        tf.debugging.assert_equal(input['order_date'], expected['order_date'])
        tf.debugging.assert_equal(label, expected['is_returning_customer'])
        tf.debugging.assert_equal(weight, expected['weight'])


def test_normalise_dates():
    from_date = to_datetime('2020-01-01')
    to_date = to_datetime('2020-01-05')

    df = DataFrame(
        {
            'timestamp': [
                to_datetime('2020-01-01'),
                to_datetime('2020-01-02'),
                to_datetime('2020-01-03'),
                to_datetime('2020-01-04'),
                to_datetime('2020-01-05'),
            ],
        }
    )

    got = normalise_dates(df.timestamp, from_date, to_date)

    expected = [0.0, 0.25, 0.5, 0.75, 1.0]

    assert list(got) == expected
