#!/usr/bin/env bash


python metagan_mnist.py --model metagan10 --n_steps 400000 --gpu_id 1
python metagan_mnist.py --model metagan10 --is_train 0 --n_steps 400000 --gpu_id 1



#python metagan_mnist.py --model metagan4
#python metagan_mnist.py --model metagan4 --is_train 0
