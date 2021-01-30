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


def encode_df(df: DataFrame, columns: List[str]) -> Tuple[DataFrame, dict]:
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


def pad_left(seq: Iterable, target_seq_len: int, padding_element=0):
    """Pad sequence with padding_elements on the left to
    target_seq_len."""

    seq = list(seq)
    seq = seq[-target_seq_len:]

    pad_len = target_seq_len - len(seq)

    return [padding_element] * pad_len + seq


def get_dataset_from_df(
    df: DataFrame, encodings: Dict[str, IntegerEncoding], seq_len: int
) -> Dataset:
    """Generate training dataset from dataframe df.

    Arguments:
      df: Dataframe with encoded categorical features by encode_df() above.
      encodings: dict of IntegerEncodings, indexed by feature names.
      seq_len: length of sequence.

    Returns:
      A tf.data.Dataset
    """

    customer_id_key = 'customer_id'
    groups = df.groupby(customer_id_key)

    for customer_id, group in groups:
        pass
    pass
