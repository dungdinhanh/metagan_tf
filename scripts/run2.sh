#!/usr/bin/env bash


python metagan_mnist.py --model metagan5 --n_steps 300000
python metagan_mnist.py --model metagan5 --is_train 0 --n_steps 300000



#python metagan_mnist.py --model metagan4
#python metagan_mnist.py --model metagan4 --is_train 0