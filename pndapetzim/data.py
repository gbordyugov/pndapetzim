from pandas import read_csv
from pandas import to_datetime


FROM_DATE = to_datetime('2015-03-01')
TO_DATE = to_datetime('2017-02-28')

ORDER_FILE_NAME = 'machine_learning_challenge_order_data.csv.gz'
LABEL_FILE_NAME = 'machine_learning_challenge_labeled_data.csv.gz'

def read_order_table(
    path='data/' + ORDER_FILE_NAME, *vargs, **kwargs
):
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
    path='data/' + LABEL_FILE_NAME, *vargs, **kwargs
):
    """Load label table. Parse customer ids as hexadecimal
    integers."""

    customer_id_key = 'customer_id'

    # Parse customer id as a hexadecimal int.
    def customer_id_converter(customer_id):
        return int(customer_id, 16)

    converters = {customer_id_key: customer_id_converter}

    return read_csv(path, converters=converters, *vargs, **kwargs)


def get_labeled_data(
    order_path='data/' + ORDER_FILE_NAME,
    label_path='data/' + LABEL_FILE_NAME,
):
    """Load order table and label table and join them on customer_id
    field."""
    customer_id_key = 'customer_id'

    orders = read_order_table(order_path)
    labels = read_label_table(label_path)

    return orders.join(
        labels.set_index(customer_id_key), on=customer_id_key, how='inner'
    )
