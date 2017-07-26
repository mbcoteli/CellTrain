#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

start cmd.exe /k "python code/create_lmdb.py"
exit