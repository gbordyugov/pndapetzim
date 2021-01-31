from __future__ import annotations  # to allow circular type annotations

from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import tensorflow as tf
from pandas import DataFrame
from pandas import read_csv
from pandas import to_datetime
from pydantic import BaseModel
from tensorflow.data import Dataset


class IntegerEncoding(BaseModel):
    """Representation of encoding for integer values.

    ix_to_value maps integer indices to integer values (such as
      restaurant ids, platform ids, etc.).
    value_to_ix maps integer values (such as restaurant ids, platform
      ids, etc.). to integer indices.
    """

    ix_to_value: Dict[int, int]
    value_to_ix: Dict[int, int]

    @property
    def vocab_size(self):
        return len(self.ix_to_value) + 1  # including zero for padding

    @staticmethod
    def fromValues(values: Iterable[int]) -> IntegerEncoding:
        """Construct an IntegerEncoding from an iterable of values."""

        ix_to_value = {}
        value_to_ix = {}

        sorted_set = sorted(set(values))

        for ix, value in enumerate(sorted_set, 1):  # zero reserved for padding
            ix_to_value[ix] = value
            value_to_ix[value] = ix

        return IntegerEncoding(
            ix_to_value=ix_to_value,
            value_to_ix=value_to_ix,
        )


CATEGORICAL_COLUMNS = [
    'restaurant_id',
    'city_id',
    'payment_id',
    'platform_id',
    'transmission_id',
]


FROM_DATE = to_datetime('2015-03-01')
TO_DATE = to_datetime('2017-02-28')

ORDER_FILE_NAME = 'machine_learning_challenge_order_data.csv.gz'
LABEL_FILE_NAME = 'machine_learning_challenge_labeled_data.csv.gz'


def read_order_table(
    path: str = 'data/' + ORDER_FILE_NAME, *vargs, **kwargs
) -> DataFrame:
    """Read order table, parsing customer ids as hexadecimal integers
    and turning order dates into 64bit timestamps."""

    customer_id_key = 'customer_id'
    order_date_key = 'order_date'

    # Parse customer id as a hexadecimal int.
    def customer_id_converter(customer_id):
        return int(customer_id, 16)

    converters = {customer_id_key: customer_id_converter}

    df = read_csv(path, converters=converters, *vargs, **kwargs)

    # Convert string dates into 64bit timestamps using a look-up
    # table, which greatly accelerates the speed of the conversion.
    dates_lut = {d: to_datetime(d) for d in df[order_date_key].unique()}
    df[order_date_key] = df[order_date_key].apply(lambda d: dates_lut[d])

    return df


def read_label_table(
    path: str = 'data/' + LABEL_FILE_NAME, *vargs, **kwargs
) -> DataFrame:
    """Load label table. Parse customer ids as hexadecimal
    integers."""

    customer_id_key = 'customer_id'

    # Parse customer id as a hexadecimal int.
    def customer_id_converter(customer_id):
        return int(customer_id, 16)

    converters = {customer_id_key: customer_id_converter}

    return read_csv(path, converters=converters, *vargs, **kwargs)


def get_labeled_data(
    order_path: str = 'data/' + ORDER_FILE_NAME,
    label_path: str = 'data/' + LABEL_FILE_NAME,
) -> DataFrame:
    """Load order table and label table and join them on customer_id
    field."""
    customer_id_key = 'customer_id'

    orders = read_order_table(order_path)
    labels = read_label_table(label_path)

    return orders.join(
        labels.set_index(customer_id_key), on=customer_id_key, how='inner'
    )


def encode_int_column(
    df: DataFrame, column_name: str
) -> Tuple[DataFrame, IntegerEncoding]:
    """Encode integer-valued column of data frame by replacing its
    values by their encodings. The zero value in encoding is reserved
    as a special padding value.

    Arguments:
      df: A dataframe containing column with `column_name` with
        integer values.
      column_name: name of the column to encode.

    Return:
      Tuple, consisting of the updated dataframe where numerical
      values in the desired column were replaced by their indices of
      the encoding, and a corresponding IntegerEncoding instance.
    """

    column = df[column_name]
    values = sorted(column.unique())
    encoding = IntegerEncoding.fromValues(values)

    df[column_name] = df[column_name].map(lambda v: encoding.value_to_ix[v])

    return df, encoding


def encode_df(
    df: DataFrame, columns: List[str] = CATEGORICAL_COLUMNS
) -> Tuple[DataFrame, dict]:
    """Encode the columns with categorical features in a dataframe
    using the function encode_int_column() above.

    Arguments:
      df: Dataframe to encode.
      columns: list of strings, columns to encode.

    Return:
      A tuple consisting of a copy of the original dataframe plus a
      dictionary mapping the names of the encoded columns to the
      corresponding IntegerEncoding.
    """
    encodings = {}

    for c in columns:
        df, encoding = encode_int_column(df, c)
        encodings[c] = encoding

    return df, encodings


def make_left_padder(target_seq_len: int, padding_element=0):
    """Return a callable that pads sequence with padding_elements on
    the left to target_seq_len."""

    padding = [padding_element] * target_seq_len

    def padder(seq):
        seq = list(seq)
        seq = seq[-target_seq_len:]
        pad_len = target_seq_len - len(seq)

        return padding[:pad_len] + seq

    return padder


def normalise_date(date, t1=FROM_DATE, t2=TO_DATE):
    """Return (date-t1)/(t2-t1) as float."""
    return (date - t1).total_seconds() / (t2 - t1).total_seconds()


def get_dataset_from_df(
    df: DataFrame,
    encodings: Dict[str, IntegerEncoding],
    seq_len: int,
    from_ts=FROM_DATE,
    to_ts=TO_DATE,
) -> Dataset:
    """Generate training dataset from dataframe df.

    Arguments:
      df: Dataframe with encoded categorical features by encode_df() above.
      encodings: dict of IntegerEncodings, indexed by feature names.
      seq_len: length of sequence.

    Returns:
      A tf.data.Dataset
    """

    action_mask_key = 'action_mask'
    customer_id_key = 'customer_id'
    order_date_key = 'order_date'
    amount_paid_key = 'amount_paid'
    label_key = 'is_returning_customer'

    padder = make_left_padder(seq_len)

    groups = df.groupby(customer_id_key)

    def generator():
        for customer_id, group in groups:
            num_actions = len(group)
            group = group.sort_values(by=order_date_key)

            amounts = group[amount_paid_key]
            dates = [
                normalise_date(d, from_ts, to_ts) for d in group[order_date_key]
            ]

            amounts = padder(amounts)
            dates = padder(dates)

            action_mask = padder([1] * num_actions)

            label = group[label_key].max()

            yield (
                {
                    action_mask_key: action_mask,
                    amount_paid_key: amounts,
                    order_date_key: dates,
                },
                int(label),
            )

    signature = (
        {
            action_mask_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
            amount_paid_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
            order_date_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    return Dataset.from_generator(generator, output_signature=signature)
