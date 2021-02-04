from __future__ import annotations  # to allow circular type annotations

from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
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


FROM_DATE = np.datetime64('2015-03-01')
TO_DATE = np.datetime64('2017-02-28')

ORDER_FILE_NAME = 'machine_learning_challenge_order_data.csv.gz'
LABEL_FILE_NAME = 'machine_learning_challenge_labeled_data.csv.gz'


def read_order_table(
    path: str = 'data/' + ORDER_FILE_NAME, *vargs, **kwargs
) -> DataFrame:
    """Read order table, parsing customer ids as hexadecimal integers
    and turning order dates into 64bit timestamps."""

    action_mask_key = 'action_mask'
    customer_id_key = 'customer_id'
    order_date_key = 'order_date'
    order_hour_cos_key = 'order_hour_cos'
    order_hour_sin_key = 'order_hour_sin'

    # Parse customer id as a hexadecimal int.
    def customer_id_converter(customer_id):
        return int(customer_id, 16)

    converters = {customer_id_key: customer_id_converter}
    dtype = {'order_hour': float}

    df = read_csv(path, converters=converters, dtype=dtype, *vargs, **kwargs)

    # Convert string dates into 64bit timestamps using a look-up
    # table, which greatly accelerates the speed of the conversion.
    dates_lut = {d: to_datetime(d) for d in df[order_date_key].unique()}
    df[order_date_key] = df[order_date_key].apply(lambda d: dates_lut[d])

    df[action_mask_key] = np.ones(len(df))
    df[order_date_key] = (df.order_date.to_numpy() - FROM_DATE) / (
        TO_DATE - FROM_DATE
    )

    angle = df.order_hour.to_numpy() / 24.0 * 2.0 * np.pi
    df[order_hour_cos_key] = np.cos(angle)
    df[order_hour_sin_key] = np.sin(angle)

    df.reset_index(inplace=True)
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
) -> Tuple[DataFrame, Dict[str, IntegerEncoding]]:
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


def pad_left(array, target_seq_len, padding_element=0):
    """Padd array with padding_element to target_seq_len on the
    left."""
    array = list(array[-target_seq_len:])
    padding = [padding_element] * (target_seq_len - len(array))
    return padding + array


def pad_left_numpy(array, target_seq_len, padding_element=0):
    """Padd array with padding_element to target_seq_len on the
    left."""
    array = array[-target_seq_len:]
    padding = np.repeat(padding_element, target_seq_len - len(array))
    return np.concatenate((padding, array))


def normalise_dates(dates, t1=FROM_DATE, t2=TO_DATE):
    """Normalise dates to the interval (t1, t2)."""

    delta = t2 - t1
    return [(d - t1) / delta for d in dates]


def normalise_dates_numpy(dates, t1=FROM_DATE, t2=TO_DATE):
    """Normalise dates to the interval (t1, t2)."""

    delta = t2 - t1
    deltas = dates - t1

    return deltas / delta


def get_dataset_from_df(
    df: DataFrame,
    seq_len: int,
    returning_weight: float = 1.0,
    from_ts=FROM_DATE,
    to_ts=TO_DATE,
) -> Dataset:
    """Generate training dataset from dataframe df.

    Arguments:
      df: Dataframe with encoded categorical features by encode_df() above.
      seq_len: length of sequence.
      returning_weight: sample weight for returning customers, used
        for over- undersampling with values larger or smaller than 1.0.
      from_ts, to_ds: are timestamps used to normalise order dates.

    Returns:
      A tf.data.Dataset
    """

    action_mask_key = 'action_mask'
    customer_id_key = 'customer_id'
    order_date_key = 'order_date'
    order_hour_cos_key = 'order_hour_cos'
    order_hour_sin_key = 'order_hour_sin'
    is_failed_key = 'is_failed'
    voucher_amount_key = 'voucher_amount'
    delivery_fee_key = 'delivery_fee'
    amount_paid_key = 'amount_paid'
    restaurant_id_key = 'restaurant_id'
    city_id_key = 'city_id'
    payment_id_key = 'payment_id'
    platform_id_key = 'platform_id'
    transmission_id_key = 'transmission_id'

    groups = df.groupby(customer_id_key, sort=False)

    def generator():
        for customer_id, group in groups:
            action_mask = pad_left(group.action_mask, seq_len)

            dates = pad_left(group.order_date, seq_len, -10.0)

            order_hour_cos = pad_left(group.order_hour_cos, seq_len)
            order_hour_sin = pad_left(group.order_hour_sin, seq_len)

            is_failed = pad_left(group.is_failed, seq_len)

            voucher_amount = pad_left(group.voucher_amount, seq_len, -1.0)
            delivery_fee = pad_left(group.delivery_fee, seq_len, -1.0)

            amount_paid = pad_left(group.amount_paid, seq_len, -1.0)

            # Categorical features
            restaurant_id = pad_left(group.restaurant_id, seq_len)
            city_id = pad_left(group.city_id, seq_len)
            payment_id = pad_left(group.payment_id, seq_len)
            platform_id = pad_left(group.platform_id, seq_len)
            transmission_id = pad_left(group.transmission_id, seq_len)

            label = int(group.is_returning_customer.max())

            weight = returning_weight if label > 0 else 1.0
            yield (
                {
                    action_mask_key: action_mask,
                    order_date_key: dates,
                    order_hour_cos_key: order_hour_cos,
                    order_hour_sin_key: order_hour_sin,
                    is_failed_key: is_failed,
                    voucher_amount_key: voucher_amount,
                    delivery_fee_key: delivery_fee,
                    amount_paid_key: amount_paid,
                    # Categorical features.
                    restaurant_id_key: restaurant_id,
                    city_id_key: city_id,
                    payment_id_key: payment_id,
                    platform_id_key: platform_id,
                    transmission_id_key: transmission_id,
                },
                [label],
                weight,
            )

    signature = (
        {
            action_mask_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
            order_date_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
            order_hour_cos_key: tf.TensorSpec(
                shape=(seq_len,), dtype=tf.float32
            ),
            order_hour_sin_key: tf.TensorSpec(
                shape=(seq_len,), dtype=tf.float32
            ),
            is_failed_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
            voucher_amount_key: tf.TensorSpec(
                shape=(seq_len,), dtype=tf.float32
            ),
            delivery_fee_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
            amount_paid_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
            # Categorical features.
            restaurant_id_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            city_id_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            payment_id_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            platform_id_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            transmission_id_key: tf.TensorSpec(
                shape=(seq_len,), dtype=tf.int32
            ),
        },
        tf.TensorSpec(shape=(1,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    return Dataset.from_generator(generator, output_signature=signature)


def load_datasets(
    order_path: str = 'data/' + ORDER_FILE_NAME,
    label_path: str = 'data/' + LABEL_FILE_NAME,
    seq_len: int = 10,
    train_ratio: int = 100,
    returning_weight: float = 1.0,
) -> Tuple[Dataset, Dataset, Dict[str, IntegerEncoding]]:
    """Load the order data and the label data, join them and return as
    a tensorflow.data.Datasets with train and testing data, plus
    encoding descriptions.

    Arguments:
      order_path, label_path: str, paths to the corresponding files.
      seq_len: int, the number of last customer orders to consider.
      train_ratio: int, how much more data is going to be put in the
        train set in comparison to the test set.
      returning_weight: float, the relative weight of samples with
        returning customers, can be seen as an "oversampling" factor.

    Returns:
      A tuple consisting of train set, test set, and a dictionary
      mapping categorical feature names to the details of their
      encoding.
    """

    df = get_labeled_data(order_path, label_path)
    df, encodings = encode_df(df)
    ds = get_dataset_from_df(df, seq_len, returning_weight).cache()

    train = ds.window(train_ratio, train_ratio + 1).flat_map(
        lambda *ds: tf.data.Dataset.zip(ds)
    )
    test = (
        ds.skip(train_ratio)
        .window(1, train_ratio + 1)
        .flat_map(lambda *ds: tf.data.Dataset.zip(ds))
    )

    return train, test, encodings
