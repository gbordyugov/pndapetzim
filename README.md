# Project Pndapetzim

## Setting up a Python environment for the project

This project uses [poetry](https://python-poetry.org/) as a build and
dependency management tool and requires a Python starting from 3.7.1.

The easiest way to install `poetry` is a) make sure that your current
`python` and `pip` executables point to a compatible Python version
and b) executing `pip install poetry`.

After cloning the repository, please run

```
poetry install
```

which will automagically creates a dedicated Python [virtual
environment](https://docs.python.org/3/tutorial/venv.html). In this
environment, `poetry` will install all dependencies of the project.

In order to use this environment, you can run any command using
`poetry run`: for example, running `pytest` with
```
poetry run pytest
```
or running JupyterLab by issuing
```
poetry run jupyter lab
```
in the command line would automatically execute its arguments in the
project environment without messing with your current Python
installation. Please note that you don't need to activate and
deactivate this environment yourself, `poetry` will handle it under
the hood upon executing a `poetry run` command like the one above. The
path to the environment is the first line of the output of `poetry
show -v`.


### Note to macos BigSur users on Intel-powered machines

`poetry` fails to install `numpy` from sources with this
configuration, you'll have to fix it by issuing
```
poetry run pip install --upgrade pip
poetry run pip install numpy=1.19.5
```
in your command line.


### Code quality

I'm supplying `test.sh` to run included unit tests, `flake8.sh` for
linting the code, `black.sh` for fixing the formatting, and `isort.sh`
for sorting the import order.


## Main findings

I build two neural networks to classify customers into returning and
non-returnin ones. The results of their training are presented in the
table below.

| Metric   | Small model | Big Model |
|:--------:|:-----------:|:---------:|
| Accuracy | 0.69        | 0.74      |
| Recall   | 0.77        | 0.71      |
| AUC      | 0.80        | 0.81      |

Those reported metrics were calculated on the test data, i.e.
customers that were not part of the training data. Those results
suggest that it would be worth it investing more time in building a
better model as a straigh-forward feature expansion didn't bring a lot
of improvement.


### The small model

The small model accepts a sequence of 20 latest customer orders
(missing orders in short histories are padded with zeros to the length
of 20), and for each order the considers only the date of order and
the order amount. It has a couple of intermediate dense layers, and
the output layer has a single output with a sigmoid activation
function.

The date of order is normalised in such a way that the timestamps of
`2015-03-01` and `2017-02-28` correspond to the normalised values of 0
and 1, respectively.


### The big model

The big model also accepts a sequence of 20 latest customer orders,
but it takes into account all available features of the orders. The
categorical features (restaurant id, city id, etc.) are encoded using
embedding layers. After that, all features for every order are
contatenated into a single vector, and the sequence of those vectors
are connected to an
[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)-like
architecture. The output of the bigger model is again a single-headed
sigmoid-activation unit.

In addition to the order date normalisation described above, for the
larger model I transform the hour of order into a pair of (x,y)
coordintes in order to account for the circular nature of hours, see,
for instance,
[here](https://en.wikipedia.org/wiki/Mean_of_circular_quantities).


### Oversampling the positive training samples

As returning customers represent approx 20% of the total number of
customers, I prescribed a relative weight of 5 to training samples
with returning customers. This resulted in a significantly better
values of the recall metric.


### Train/test data split

For the training and evaluation purposes, all data was split in such
way that every 101st customer went into the test data, and the rest of
the customers constituted the training data.


## Exploratory data analysis

As the data is in the CSV format, I decided to conduct the inital data
analysis steps using the standard command line tools (you might need
to install
[gzcat](https://www.freebsd.org/cgi/man.cgi?query=gzcat&sektion=1&n=1)).
Let us see what we can learn about the data.

This is the total number of orders
```
➜  data git:(main) ✗ gzcat machine_learning_challenge_order_data.csv.gz | wc -l
  786601
```

This is the total number of unique customer ids in the _order data_:
```
➜  data git:(main) ✗ gzcat machine_learning_challenge_order_data.csv.gz | cut -d, -f 1 | sort | uniq | wc -l
  245456
```

Seems to be the same as the number of customers in the _labelled data_:
```
➜  data git:(main) ✗ gzcat machine_learning_challenge_labeled_data.csv.gz | cut -d, -f 1 | sort | uniq | wc -l
  245456
```

The earliest and the latest order dates are:
```
➜  data git:(main) ✗ gzcat machine_learning_challenge_order_data.csv.gz | cut -d, -f2 | sort | head -n 1
2012-05-17
➜  data git:(main) ✗ gzcat machine_learning_challenge_order_data.csv.gz | cut -d, -f2 | sort | tail -n 2 | head -n 1
2017-02-27
```

The number of returning and non-returning customers is:
```
➜  data git:(main) ✗ gzcat machine_learning_challenge_labeled_data.csv.gz | cut -d, -f2 | sort | uniq -c | head -n 2
189948 0
55507 1
```

The number of different restaurant ids:
```
➜  data git:(main) ✗ gzcat machine_learning_challenge_order_data.csv.gz | cut -d, -f9 | sort | uniq -c | wc -l
   13570
```

The most frequent restaurant ids are:
```
➜  data git:(main) ✗ gzcat machine_learning_challenge_order_data.csv.gz | cut -d, -f9 | sort | uniq -c | sort -n | tail
 882 29593498
 918 30633498
 922 18603498
 935 105253498
 942 146723498
 967 88773498
 999 154543498
1031 192673498
1071 983498
1317 37623498
```

The same analysis can be applied to city id, payment id, and so on by
choosing a different field index: `-f10` would correspond to city id,
`-f11` to payment id, etc.


## Data transformation before training


### Optimising data types

I mapped user ids from strings to integers by parsing them as
hexadecimal numbers for faster join and groupBy operations on the
dataframe, see
[here](https://github.com/gbordyugov/pndapetzim/blob/main/pndapetzim/data.py#L81)
and
[here](https://github.com/gbordyugov/pndapetzim/blob/main/pndapetzim/data.py#L114).

I
[mapped](https://github.com/gbordyugov/pndapetzim/blob/main/pndapetzim/data.py#L97)
string order dates to float64 timestamps for a smaller memory
footprint of the dataframe.


### Getting labelled training data

I join the order table with the label table on the `customer_id`
field by an inner join, see the [source
code](https://github.com/gbordyugov/pndapetzim/blob/main/pndapetzim/data.py#L122).

I normalise the order date in such a way that the resulting date value
are between 0 and 1, 0 meaning the stated beginning of the time window
2015-03-01 and 1 corresponding to its end 2017-02-28, see
[code](https://github.com/gbordyugov/pndapetzim/blob/main/pndapetzim/data.py#L95).

The order hour is
[transformed](https://github.com/gbordyugov/pndapetzim/blob/main/pndapetzim/data.py#L97)
into a pair of cartesian coordinates representing the hour on the unit
circle.
