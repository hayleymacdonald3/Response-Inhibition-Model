#!/bin/bash

# Usage: ./profile.sh Go_Trials.py

# If on Windows, run this on the command line and replace $1 with the name of the target script
python -m cProfile -o profile.data $1
