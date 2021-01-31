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
