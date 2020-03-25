#!/usr/bin/env bash


python metagan_mnist.py --model metagan8 --n_steps 300000
python metagan_mnist.py --model metagan8 --is_train 0 --n_steps 300000


#python metagan_mnist.py --model metagan2 --n_steps 300000
#python metagan_mnist.py --model metagan2 --is_train 0