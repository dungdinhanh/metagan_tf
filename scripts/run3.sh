#!/usr/bin/env bash




python metagan_mnist_nogradgen.py --model metagannge1 --n_steps 400000 --gpu_id 0 --lambda_ent 0.1
python metagan_mnist_nogradgen.py --model metagannge1 --is_train 0 --n_steps 400000 --gpu_id 0 --lambda_ent 0.1
python metagan_mnist_cp.py --model metagange1 --n_steps 400000 --gpu_id 1 --lambda_ent 0.1

#python metagan_mnist.py --model metagan2 --n_steps 400000 --gpu_id 0 --lambda_ent 0.4
#python metagan_mnist.py --model metagan2 --is_train 0 --n_steps 400000 --gpu_id 0 --lambda_ent 0.1
#python metagan_mnist_cp.py --model metagan2 --n_steps 400000 --gpu_id 0 --lambda_ent 0.1
#
#python metagan_mnist.py --model metagan3_lambda0.4 --n_steps 400000 --gpu_id 0 --lambda_ent 0.4
#python metagan_mnist.py --model metagan3_lambda0.4 --is_train 0 --n_steps 400000 --gpu_id 0 --lambda_ent 0.1
#python metagan_mnist_cp.py --model metagan2 --n_steps 400000 --gpu_id 0 --lambda_ent 0.1

#python metagan_mnist.py --model metagan4
#python metagan_mnist.py --model metagan4 --is_train 0
