export CUDA_VISIBLE_DEVICES=0

python examples/adversarial/main.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet