import os
import sys
import time
import subprocess
import re
import argparse

parser = argparse.ArgumentParser(
    description='Use specific tool to train baseline models')
parser.add_argument('--all', dest = 'all', default = True, type = bool, help = 'run all the baseline models training or not (true or false), default: false')
parser.add_argument('--tool', dest = 'tool', default = 'dl2', type =str, help = 'specify which tool to train the baseline: (dl2, deeprepair, apricot')
parser.add_argument('--net_type', dest='nettype', default='resnet', type=str,
                    help='networktype: resnet, mnist, and fmnist')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, mnist, fmnist')
parser.add_argument('--additional_param', dest ='param', default = '', type =str, help='if you want to set the parameters manually, add this')
parser.add_argument(
    '--saved_path', default='/set/your/model/path', type=str, metavar='PATH')

args = parser.parse_args()

#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "RobustnessT(eps1=13.8, eps2=0.9)" --report-dir reports
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "RobustnessG(eps=0.3, delta=0.52)" --report-dir reports --num-iters 7
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "LipschitzT(L=1.0)" --report-dir reports
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "LipschitzG(L=1.0, eps=0.3)" --report-dir reports --num-iters 5
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "CSimilarityT(delta=0.01)" --report-dir reports
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "CSimilarityG(delta=1.0, eps=0.3)" --report-dir reports --num-iters 10

def train_baseline(tool, dataset, nettype, savepath, additional_param):
    if tool =='dl2':
        if additional_param == "":
            cmd = "python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset " + dataset \
                + "--constraint \"RobustnessT(eps1=13.8, eps2=0.9)\" --report-dir reports" + " --saved_model " \
                    + savepath + "train_baseline_bydl2_" + dataset + "_" + nettype + ".pt"
            with open("stdout.txt","wb") as out, open("stderr.txt","wb") as err:
                ex=subprocess.Popen(cmd,stdout = subprocess.PIPE,stdin = subprocess.PIPE)
                while ex.poll() is None:
                    line=ex.stdout.readline().decode("utf8")
                    print(line)
                status = ex.wait()
        else:
            cmd = "python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset " + dataset \
                + "--constraint \"RobustnessT(eps1=13.8, eps2=0.9)\" --report-dir reports" + " --saved_model " \
                    + savepath + "train_baseline_bydl2_" + dataset + "_" + nettype + ".pt"
def main():
    if args.all =="":
        train_baseline("dl2", "cifar10", "resnet", "./pretrained_models/dl2")
        train_baseline("dl2", "cifar100", "resnet", "./pretrained_models/dl2")
        train_baseline("dl2", "mnist", "mnist", "./pretrained_models/dl2")
        train_baseline("dl2", "fmnisy", "fmnist", "./pretrained_models/dl2")
        
    else:
        train_baseline(args.tool, args.dataset, args.nettype, args.saved_path, args.additional_param)
if __name__ == '__main__':
    main()


