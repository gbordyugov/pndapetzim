from pandas import read_csv
from pandas import to_datetime


def read_order_table(
    path='data/machine_learning_challenge_order_data.csv.gz', *vargs, **kwargs
):
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
