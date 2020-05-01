#!/usr/bin/env bash


#python metagan_mnist_test.py --model metagan_test_nofakeaux2 --n_steps 400000
#python metagan_mnist_test.py --model metagan_test_nofakeaux2 --is_train 0 --n_steps 400000

python metagan_mnist_test.py --model metagan1_nofaux --n_steps 400000 --lambda_ent 0.6
python metagan_mnist_test.py --model metagan1_nofaux --is_train 0 --n_steps 400000 --lambda_ent 0.6


python metagan_mnist_test.py --model metagan2_nofaux --n_steps 400000 --lambda_ent 0.6
python metagan_mnist_test.py --model metagan2_nofaux --is_train 0 --n_steps 400000 --lambda_ent 0.6


python metagan_mnist_test.py --model metagan3_nofaux --n_steps 400000 --lambda_ent 0.6
python metagan_mnist_test.py --model metagan3_nofaux --is_train 0 --n_steps 400000 --lambda_ent 0.6


python metagan_mnist_test.py --model metagan3_nofaux --n_steps 400000 --lambda_ent 0.4
python metagan_mnist_test.py --model metagan3_nofaux --is_train 0 --n_steps 400000 --lambda_ent 0.4

python metagan_mnist_test.py --model metagan3_nofaux --n_steps 400000 --lambda_ent 0.2
python metagan_mnist_test.py --model metagan3_nofaux --is_train 0 --n_steps 400000 --lambda_ent 0.2

#python metagan_mnist.py --model metagan11 --n_steps 400000
#python metagan_mnist.py --model metagan11 --is_train 0 --n_steps 400000

