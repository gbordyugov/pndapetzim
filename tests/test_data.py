from pandas import DataFrame
from pndapetzim.data import encode_df
from pndapetzim.data import encode_int_column
from pndapetzim.data import IntegerEncoding


def test_integer_encoding():
    values = [18, 13, 16, 18, 19]
    expected = IntegerEncoding(
        ix_to_value={1: 13, 2: 16, 3: 18, 4: 19},
        value_to_ix={13: 1, 16: 2, 18: 3, 19: 4},
    )
    got = IntegerEncoding.fromValues(values)

    assert got == expected


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
