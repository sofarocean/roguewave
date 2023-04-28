#!/bin/bash -e
python setup.py sdist bdist_wheel

# For production PyPi:
twine upload dist/*
