#!/usr/bin/env bash


python dcgan_mnist.py --model dcgan1 --n_steps 400000
python dcgan_mnist.py --model dcgan1 --is_train 0 --n_steps 400000

python dcgan_mnist.py --model dcgan2 --n_steps 400000
python dcgan_mnist.py --model dcgan2 --is_train 0 --n_steps 400000

python dcgan_mnist.py --model dcgan3 --n_steps 400000
python dcgan_mnist.py --model dcgan3 --is_train 0 --n_steps 400000

python dcgan_mnist.py --model dcgan4 --n_steps 400000
python dcgan_mnist.py --model dcgan4 --is_train 0 --n_steps 400000

python dcgan_mnist.py --model dcgan5 --n_steps 400000
python dcgan_mnist.py --model dcgan5 --is_train 0 --n_steps 400000
#python metagan_mnist.py --model metagan4
#python metagan_mnist.py --model metagan4 --is_train 0