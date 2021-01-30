#!/bin/sh

poetry run python -m flake8 --max-line-length 80 $* .
