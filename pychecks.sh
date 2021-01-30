#!/bin/sh

set -e

./black.sh --diff --check
./flake8.sh
./isort.sh --diff --check-only
