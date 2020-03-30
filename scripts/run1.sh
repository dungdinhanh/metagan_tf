#!/usr/bin/env bash


#python metagan_mnist_test.py --model metagan_test_nofakeaux2 --n_steps 400000
#python metagan_mnist_test.py --model metagan_test_nofakeaux2 --is_train 0 --n_steps 400000

python metagan_mnist.py --model metagan1_lambda0.2 --n_steps 400000
python metagan_mnist.py --model metagan1_lambda0.2 --is_train 0 --n_steps 400000


python metagan_mnist.py --model metagan2_lambda0.2 --n_steps 400000
python metagan_mnist.py --model metagan2_lambda0.2 --is_train 0 --n_steps 400000


python metagan_mnist.py --model metagan3_lambda0.2 --n_steps 400000
python metagan_mnist.py --model metagan3_lambda0.2 --is_train 0 --n_steps 400000


#python metagan_mnist.py --model metagan11 --n_steps 400000
#python metagan_mnist.py --model metagan11 --is_train 0 --n_steps 400000

