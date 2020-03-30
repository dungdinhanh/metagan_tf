#!/usr/bin/env bash


python dcgan_mnist.py --model dcgan6 --n_steps 400000
python dcgan_mnist.py --model dcgan6 --is_train 0 --n_steps 400000

python dcgan_mnist.py --model dcgan7 --n_steps 400000
python dcgan_mnist.py --model dcgan7 --is_train 0 --n_steps 400000

python dcgan_mnist.py --model dcgan8 --n_steps 400000
python dcgan_mnist.py --model dcgan8 --is_train 0 --n_steps 400000

python dcgan_mnist.py --model dcgan9 --n_steps 400000
python dcgan_mnist.py --model dcgan9 --is_train 0 --n_steps 400000

python dcgan_mnist.py --model dcgan10 --n_steps 400000
python dcgan_mnist.py --model dcgan10 --is_train 0 --n_steps 400000
#python metagan_mnist.py --model metgan4
#python metagan_mnist.py --model metagan4 --is_train 0

