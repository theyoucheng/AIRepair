from inspect import ArgSpec
import os
import sys
import time
import subprocess
import re
import sh
import argparse
import subprocess
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import torch
import numpy as np
import torch.optim as optim
import json
import time
from torchvision import datasets, transforms
import torch.onnx
import onnx
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from resnet import ResNet50, ResNet18, ResNet34
from models import MLP, MnistNet



parser = argparse.ArgumentParser(
    description='Compare different repairing tools')

parser.add_argument('--net_type', dest='nettype', default='resnet', type=str,
                    help='networktype: resnet, mnist, and fmnist')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, mnist, fmnist')
parser.add_argument(
    '--pretrained', dest= 'pretrained', default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument('--depth', default=18, type=int,
                    help='depth of the network (default: 18)')

class GPUManager():
    """GPU manager, keeps track which GPUs used and which are avaiable."""

    def __init__(self, available_gpus):
        """Initialize GPU manager.
        Args:
            available_gpus: GPU ids of gpus that can be used.
        """
        self.semaphore = threading.BoundedSemaphore(len(available_gpus))
        self.gpu_list_lock = threading.Lock()
        self.available_gpus = list(available_gpus)

    def get_gpu(self):
        """Get a GPU, if none is available, wait until there is."""
        self.semaphore.acquire()
        with self.gpu_list_lock:
            gpu = self.available_gpus.pop()
        return GPU(gpu, self)

    def release_gpu(self, gpu):
        """Relesae a GPU after use.
        Args:
            gpu: GPU object returned from get_gpu.
        """
        gpu = gpu.nr
        with self.gpu_list_lock:
            self.available_gpus.append(gpu)
            self.semaphore.release()


class GPU():
    """Representation of a GPU."""

    def __init__(self, nr, manager):
        """Set up GPU Object.
        Args:
            nr: Which GPU id the GPU has
            manager: The manager of the GPU
        """
        self.nr = nr
        self.manager = manager

    def release(self):
        """Release the GPU."""
        self.manager.release_gpu(self)

    def __str__(self):
        """Return string representation of GPU."""
        return str(self.nr)


def run_command_with_gpu(command, gpu):
    """Run command using Popen, set CUDA_VISIBLE_DEVICE and free GPU when done.
    Args:
        command: string with command
        gpu: GPU object
    Returns: Thread object of the started thread.
    """
    myenv = os.environ.copy()
    myenv['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print(f'Processing command `{command}` on gpu {gpu}')

    def run_then_release_GPU(command, gpu):
        myenv = os.environ.copy()
        myenv['CUDA_VISIBLE_DEVICES'] = str(gpu)
        proc = subprocess.Popen(
            args=command,
            shell=True,
            env=myenv
        )
        proc.wait()
        gpu.release()
        return

    thread = threading.Thread(
        target=run_then_release_GPU,
        args=(command, gpu)
    )
    thread.start()
    # returns immediately after the thread starts
    return thread

def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory

def narrow_setup(interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    while gpu_memory > 1000 or gpu_power > 20:  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + cmd)
    os.system(cmd)

def watch_GPU(GPU_free=0.):
    # device
    deviceCount = pynvml.nvmlDeviceGetCount()
    print('GPU number count', deviceCount)
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        print('GPU %d is :%s' % (i, gpu_name))
        
        memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print("GPU %d Memory Total: %.4f G" % (i, memo_info.total / 1024 / 1024 / 1000))
        GPU_free += memo_info.free / 1024 / 1024
        print("GPU %d Memory Free: %.4f G" % (i, memo_info.free / 1024 / 1024 / 1000))
        print("GPU %d Memory Used: %.4f G" % (i, memo_info.used / 1024 / 1024 / 1000))

        Temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
        print("Temperature is %.1f C" % (Temperature))

        speed = pynvml.nvmlDeviceGetFanSpeed(handle)
        print("Fan speed is ", speed)

        power_ststus = pynvml.nvmlDeviceGetPowerState(handle)
        print("Power ststus", power_ststus)
        return GPU_free
    
def LastNlines(fname, N):     
    assert N >= 0
    pos = N + 1
    lines = []
    with open(fname) as f:
        while len(lines) <= N:
            try:
                f.seek(-pos, 2)
            except IOError:
                f.seek(0)
                break
            finally:
                lines = list(f)
            pos *= 2
    return lines[-N:]

def check_model_dl2(dataset, net_type, resume_from):
    # python main.py --net_type resnet --dataset cifar100 --resume_from cifar100_baseline_resnet50_semi_dl2_1600
    cmd = "python evaluate_model_dl2.py --net_type" + net_type + " --dataset " + dataset + " --resume_from " + resume_from
    with open("stdout.txt","wb") as out, open("stderr.txt","wb") as err:
        subprocess.Popen(cmd,stdout=out,stderr=err)
    foundresult = False
    N = 5
    fname="stdout.txt"
    try:
        lines = LastNlines(fname, N)
        for line in lines:
            resultline = " "
            if "Test Result" in line:
                foundresult = True
            elif "CUDA out of memory" in line:
                foundresult = False
                print('Runtime Error: cuda out of memory')
                return none
            elif "Error" in line:
                foundresult = False
                print('Unknown Error, cannot found the result in the log')
                return none
            if foundresult and "Acc@1" in line:
                ACC_1 = re.findall("\d+\.\d+", line)
            elif foundresult and "CAcc" in line:
                CACC = re.findall("\d+\.\d+", line)
            elif foundresult and "GroupACC" in line:
                GroupACC = re.findall("\d+\.\d+", line)
    except:
        print('File not found')
    print("This model accuracy is: "+ model_accuracy + " CACC is: " + model_CACC + " Group Acc is:" + G_ACC)
    return ACC_1, CACC, GroupACC

def check_model_deeprepair(filepath):
    #python check_evaluate_model.py  --net_type resnet --dataset cifar10 --batch_size 256 --workers 4 --lr 0.1 --depth 18 --pretrained ./CSimilarityT.pt --checkmodel
    cmd = "python evaluate_model_dl2.py --net_type" + net_type + " --dataset " + dataset + " --resume_from " + resume_from
    with open("stdout.txt","wb") as out, open("stderr.txt","wb") as err:
        subprocess.Popen(cmd,stdout=out,stderr=err)
    foundresult = False
    result = False
    N = 10
    try:
        lines = LastNlines(fname, N)
        for line in lines:
            resultline = " "
            if "Epoch: [-1/60]" in line:
                foundresult = True
                acc = re.search("^[1-9]\d*\.\d*|0\.\d*[1-9]\d*$", line)
            elif "CUDA out of memory" in line:
                foundresult = False
                print('Runtime Error: cuda out of memory')
                return none
            elif "Error" in line:
                foundresult = False
                print('Unknown Error, cannot found the result in the log')
                return none                
            if foundresult and "pair accuracy" in line:
                confusion_acc = re.findall("^[1-9]\d*\.\d*|0\.\d*[1-9]\d*$", line)
    except:
        print('File not found')
    print("This model accuracy is: "+ model_accuracy + " CACC is: " + model_CACC + " Group Acc is:" + G_ACC)
    return acc, confusion_acc

def convert_pth_to_pt(net_type, depth, dataset, inputpath, outputpath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if net_type == 'resnet':
        if depth == 18:
            if dataset == 'cifar10':
                trained_model = ResNet18(10)
            elif dataset == 'cifar100':
                trained_model = ResNet18(100)
        if depth == 34:
            if dataset == 'cifar10':
                trained_model = ResNet34(10)
            elif dataset == 'cifar100':
                trained_model = ResNet34(100)
        if depth == 50:
            if dataset == 'cifar10':
                trained_model = ResNet50(10)
            elif dataset == 'cifar100':
                trained_model = ResNet50(10)          
    elif net_type == 'mnist':
        trained_model = MnistNet()
    elif net_type == 'fmnist':
        trained_model = MnistNet()
       
    trained_model.load_state_dict(torch.load(input_path),strict= False)
    trained_model.to(device)
    trained_model.eval()
    torch.save(trained_model,outputpath)

    

def repair_using_deeprepair(dataset, net_type, depth, expname, pre_trained, saved_path):
    #nohup python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar100 --depth 50 --batch_size 128 --lr 0.1 --expname cifar100_resnet50_2_4 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet50_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 0.9 > cifar100_repair_newbn_ratio0.9_resnet50.log 2>&1 &
    #run different repairing method then pick the best one
    #first run confusion_exp_newbn
    cmd = "python3 repair_confusion_exp_newbn.py --net_type" + net_type + " --dataset " + dataset + " --depth " + depth + "  --batch_size 128 --lr 0.1 --expname" + expname + "--epoch 60 --beta 1.0 --cutmix_prob 0 --pretrained " + pretrained + " --lam 0 --extra 128 --replace --ratio 0.9"
    with open("stdout_confusion_exp_newbn.txt","wb") as out, open("stderr_confusion_exp_newbn.txt","wb") as err:
        subprocess.Popen(cmd,stdout=out,stderr=err)

    

def repair_using_dl2(dataset, net_type, depth, pre_trained, saved_path):
    #python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0.04 --dataset cifar10  --constraint "RobustnessT(eps1=13.8, eps2=0.9)" --report-dir reports


    
def repair_using_apricot(dataset, net_type, depth, pre_trained, saved_path):
    #python weight_adjust_mnist_fmnist.py -err_src 3 -err_dst 5
    cmd = "the command to run apricot benchmark"
    if dataset == 'cifar10':
        if net_type == 'resnet18':
            cmd = "python weight_adjust_mnist_fmnist.py -err_src 3 -err_dst 5"
        if net_type == 'resnet34':




def main():
    print("Now run benchmark using Apricot, DeepRepair and DL2")
    print("Firstly check the model")
    deeprepair_acc, deeprepair_con_acc = check_model_deeprepair(ArgSpec.pretrained)
    dl2_acc, dl2_cacc, dl2_group_acc = check_model_dl2(ArgSpec.pretrained)
    print("check model using deeprepair")
    print("The acc of the model is " + deeprepair_acc + ", and the confusion_acc of the model is" + deeprepair_acc)
    print("The acc of the model (check by dl2) is " + dl2_acc + ", and the cacc is " + dl2_cacc + "and the group_acc is " + dl2_group_acc)
    print("Then apply repair:")
    print("Firstly run apricot")
    
    repair_using_apricot(args.dataset, args.nettype, 18, args.pretrained, "./model/repaired_by_apricot")
    
    

if __name__ == '__main__':
    main()


    
  
                




