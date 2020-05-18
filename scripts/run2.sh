#!/usr/bin/env bash




python metagan_mnist.py --model metagan1_lambda0.4 --n_steps 400000 --gpu_id 1 --lambda_ent 0.4
python metagan_mnist.py --model metagan1_lambda0.4 --is_train 0 --n_steps 400000 --gpu_id 1 --lambda_ent 0.4


#python metagan_mnist.py --model metagan2_lambda0.4 --n_steps 400000 --gpu_id 1 --lambda_ent 0.4
#python metagan_mnist.py --model metagan2_lambda0.4 --is_train 0 --n_steps 400000 --gpu_id 1 --lambda_ent 0.4
#
#
#python metagan_mnist.py --model metagan3_lambda0.4 --n_steps 400000 --gpu_id 1 --lambda_ent 0.4
#python metagan_mnist.py --model metagan3_lambda0.4 --is_train 0 --n_steps 400000 --gpu_id 1 --lambda_ent 0.4


#python metagan_mnist.py --model metagan4
#python metagan_mnist.py --model metagan4 --is_train 0
