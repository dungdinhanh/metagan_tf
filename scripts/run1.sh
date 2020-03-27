#!/usr/bin/env bash


python metagan_mnist.py --model metagan_test_nofakeaux1 --n_steps 400000
python metagan_mnist.py --model metagan_test_nofakeaux1 --is_train 0 --n_steps 400000



#python metagan_mnist.py --model metagan11 --n_steps 400000
#python metagan_mnist.py --model metagan11 --is_train 0 --n_steps 400000