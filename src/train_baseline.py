import os
import sys
import time
import subprocess
import re
import sh
import argparse

parser = argparse.ArgumentParser(
    description='Use specific tool to train baseline models')

parser.add_argument('--net_type', dest='nettype', default='pyramidnet', type=str,
                    help='networktype: resnet, mnist, and fmnist')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, mnist, fmnist')
parser.add_argument(
    '--saved_path', default='/set/your/model/path', type=str, metavar='PATH')
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "RobustnessT(eps1=13.8, eps2=0.9)" --report-dir reports
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "RobustnessG(eps=0.3, delta=0.52)" --report-dir reports --num-iters 7
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "LipschitzT(L=1.0)" --report-dir reports
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "LipschitzG(L=1.0, eps=0.3)" --report-dir reports --num-iters 5
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "CSimilarityT(delta=0.01)" --report-dir reports
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "CSimilarityG(delta=1.0, eps=0.3)" --report-dir reports --num-iters 10

def train_baseline(tool, dataset, model, savepath):
    return acc
