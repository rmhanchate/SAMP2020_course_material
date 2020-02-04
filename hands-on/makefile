#!/usr/bin/env bash

venv:. venv/bin/activate  # Activate the virtualenv created

venv/bin/activate: requirements.txt  # Look into requirements.txt for necessary dependencies
	test -d venv || virtualenv venv  # Test whether virtualenv named venv exists, else create a new one
	. venv/bin/activate; pip install -Ur requirements.txt  #  Create the virtualenv and install dependencies using pip
	touch venv/bin/activate  # Activate the virtualenv

test: venv
	. venv/bin/activate;nosetests project/test

