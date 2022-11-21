from multiprocessing import freeze_support
import os
import sys
import time
import subprocess
import re
import argparse

parser = argparse.ArgumentParser(
    description='Use specific tool to train baseline models')
parser.add_argument('--all', default='False', action='store_true', help = 'run all repairing on the pretrained models, default: false')
parser.add_argument('--tool', dest = 'tool', default = 'dl2', type =str, help = 'specify which tool to train the baseline: (dl2, deeprepair, apricot')
parser.add_argument('--net_type', dest='nettype', default='resnet', type=str,
                    help='networktype: resnet, mnist, and fmnist')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, mnist, fmnist')
parser.add_argument('--additional_param', dest ='additional_param', default = '', type =str, help='if you want to set the parameters manually, add this')
parser.add_argument(
    '--saved_path', default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument('--depth', default=18, type=int,
                    help='depth of the network (default: 18)')

args = parser.parse_args()

#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "RobustnessT(eps1=13.8, eps2=0.9)" --report-dir reports
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "RobustnessG(eps=0.3, delta=0.52)" --report-dir reports --num-iters 7
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "LipschitzT(L=1.0)" --report-dir reports
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "LipschitzG(L=1.0, eps=0.3)" --report-dir reports --num-iters 5
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "CSimilarityT(delta=0.01)" --report-dir reports
#python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "CSimilarityG(delta=1.0, eps=0.3)" --report-dir reports --num-iters 10
#python3 train_baseline.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --expname DeepInspect_1 --epochs 300 --beta 1.0 --cutmix_prob 1

def train_baseline(tool, dataset, nettype, depth, savepath, additional_param):
    if tool =='dl2':
        if additional_param == "":
            cmd = "python patch_dl2_main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset " + dataset \
                + " --constraint \"RobustnessT(eps1=13.8, eps2=0.9)\" --report-dir reports" + " --saved_model " \
                    + savepath + "train_baseline_bydl2_" + dataset + "_" + nettype + ".pt"
            print(cmd)
            with open("train_baseline_dl2_stdout.txt","wb") as out, open("train_baseline_dl2_stderr.txt","wb") as err:
                ex=subprocess.Popen(cmd,stdout = subprocess.PIPE,stdin = subprocess.PIPE)
                while ex.poll() is None:
                    line=ex.stdout.readline().decode("utf8")
                    if line != "":
                        print(line)
                status = ex.wait()
        else:
            cmd = "python patch_dl2_main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset " + dataset \
                + " --constraint \"RobustnessT(eps1=13.8, eps2=0.9)\" --report-dir reports" + " --saved_model " \
                    + savepath + "plus_additional_train_baseline_bydl2_" + dataset + "_" + nettype + ".pt" + " " + additional_param
            with open("train_baseline_dl2_stdout_plus_additional_param.txt","wb") as out, open("train_baseline_dl2_stderr_plus_additional_param.txt","wb") as err:
                ex=subprocess.Popen(cmd,stdout = subprocess.PIPE,stdin = subprocess.PIPE)
                while ex.poll() is None:
                    line=ex.stdout.readline().decode("utf8")
                    if line != "":
                        print(line)
                status = ex.wait()
    elif tool =='deeprepair':
        if additional_param == "":
            cmd = "python train_baseline.py --net_type " + nettype + " --dataset " + dataset + " --depth " + str(depth) + " --batch_size 256 --lr 0.1 --expname " \
                + dataset + "_" + nettype + str(depth) + "_trained_by_" + tool + " --epochs 300 --beta 1.0 --cutmix_prob 1"
            with open("trained_by_deeprepair_stdout.txt","wb") as out, open("trained_by_deeprepair_stderr.txt","wb") as err:
                ex=subprocess.Popen(cmd,stdout = subprocess.PIPE,stdin = subprocess.PIPE)
                while ex.poll() is None:
                    line=ex.stdout.readline().decode("utf8")
                    if line != "":
                        print(line)
                status = ex.wait()
        else:
            cmd = "python train_baseline.py --nettype " + nettype + " --dataset " +dataset + " --depth " + str(depth) + " --expname " +\
                dataset + "_" + nettype + str(depth) + "_plus_additional_trained_by_" + tool + " " + additional_param
            with open("trained_by_deeprepair_stdout_plus_additional_param.txt","wb") as out, open("trained_by_deeprepair_stderr_plus_additional_param.txt","wb") as err:
                ex=subprocess.Popen(cmd,stdout = subprocess.PIPE,stdin = subprocess.PIPE)
                while ex.poll() is None:
                    line=ex.stdout.readline().decode("utf8")
                    if line != "":
                        print(line)
                status = ex.wait()

    elif tool == 'apricot':
        if additional_param =="":
            cmd = "mkdir weights rDLM_weights"
            os.system(cmd)
            cmd = "python iDLM_train.py"
            with open("train_iDLM_apricot_stdout.txt","wb") as out, open("train_iDLM_apricot_stderr.txt","wb") as err:
                ex=subprocess.Popen(cmd,stdout = subprocess.PIPE,stdin = subprocess.PIPE)
                while ex.poll() is None:
                    line=ex.stdout.readline().decode("utf8")
                    if line != "":
                        print(line)
                status = ex.wait()
            cmd = "python rDLM_train.py"
            with open("train_rDLM_apricot_stdout.txt","wb") as out, open("train_rDLM_apricot_stderr.txt","wb") as err:
                ex=subprocess.Popen(cmd,stdout = subprocess.PIPE,stdin = subprocess.PIPE)
                while ex.poll() is None:
                    line=ex.stdout.readline().decode("utf8")
                    if line != "":
                        print(line)
                status = ex.wait()
            
def main():
    if args.all =="":
        train_baseline("dl2", "cifar10", "resnet", "./pretrained_models/dl2","")
        train_baseline("dl2", "cifar100", "resnet", "./pretrained_models/dl2","")
        train_baseline("dl2", "mnist", "mnist", "./pretrained_models/dl2","")
        train_baseline("dl2", "fmnist", "fmnist", "./pretrained_models/dl2","")
        
    else:
        train_baseline(args.tool, args.dataset, args.nettype, args.depth, args.saved_path, args.additional_param)
if __name__ == '__main__':
    freeze_support()
    main()


