import os
import time
from subprocess import PIPE, run

def execute(command):
    result = run(command, universal_newlines=True, shell=True, capture_output=True, text=True)
    # print(result.stdout)
    print(result.stderr)


method_properties = {
'w-aug': {
    "param_name": "weight",
    "filename": "exp_oversampling",
    "replace": "",
    "batch_size": 128,
    "extra_batch_size": 0,
},
'w-bn': {
    "param_name": "ratio",
    "filename": "exp_newbn",
    "replace": " --replace",
    "batch_size": 128,
    "extra_batch_size": 128,
},
'w-loss': {
    "param_name": "target_weight",
    "filename": "exp_weighted_loss",
    "replace": "",
    "batch_size": 128,
    "extra_batch_size": 10,
},
'w-dbr': {
    "param_name": "lam",
    "filename": "dbr",
    "replace": "",
    "batch_size": 128,
    "extra_batch_size": 10,
}
}



dataset_model_classes = [('cifar10', 'resnet-18', (3, 5, 2)), ('cifar10', 'vggbn-11', (3, 5, 2)), ('cifar10', 'mobilenetv2-115', (3, 5, 2)), ('cifar100', 'resnet-34', (98, 35, 11)), ('cifar10_multipair', 'resnet-18', (3, 5, 1, 9))]
tasks = ['confusion', 'bias']
methods = ['w-aug', 'w-bn', 'w-loss', 'w-dbr']
params = [0.1, 0.3, 0.5, 0.7, 0.9]
epochs = 60
verbose = ""

dataset_model_classes = [('cifar10', 'mobilenetv2-115', (3, 5, 2))]
tasks = ['confusion', 'bias']
methods = ['w-aug', 'w-bn', 'w-loss', 'w-dbr']
params = [0.1, 0.3, 0.5, 0.7, 0.9]
# epochs = 2


# config_list = [
# ('cifar10', 'resnet-18', (3, 5, 2), 'confusion', 'w-aug', 1.0),
# ('cifar10', 'vggbn-11', (3, 5, 2), 'confusion', 'w-aug', 1.0),
# ('cifar10', 'mobilenetv2-115', (3, 5, 2), 'confusion', 'w-aug', 1.0),
# ('cifar100', 'resnet-34', (98, 35, 11), 'confusion', 'w-aug', 1.0),
#
# ('cifar10', 'resnet-18', (3, 5, 2), 'confusion', 'w-aug', 0.7),
# ('cifar10', 'resnet-18', (3, 5, 2), 'confusion', 'w-bn', 0.7),
# ('cifar10', 'resnet-18', (3, 5, 2), 'confusion', 'w-bn', 0.9),
# ('cifar10', 'resnet-18', (3, 5, 2), 'confusion', 'w-loss', 0.9),
# ('cifar10', 'resnet-18', (3, 5, 2), 'confusion', 'w-dbr', 0.9),
#
# ('cifar10', 'vggbn-11', (3, 5, 2), 'confusion', 'w-aug', 0.9),
# ('cifar10', 'vggbn-11', (3, 5, 2), 'confusion', 'w-bn', 0.7),
# ('cifar10', 'vggbn-11', (3, 5, 2), 'confusion', 'w-loss', 0.9),
# ('cifar10', 'vggbn-11', (3, 5, 2), 'confusion', 'w-dbr', 0.1),
#
# ('cifar10', 'mobilenetv2-115', (3, 5, 2), 'confusion', 'w-aug', 0.5),
# ('cifar10', 'mobilenetv2-115', (3, 5, 2), 'confusion', 'w-bn', 0.7),
# ('cifar10', 'mobilenetv2-115', (3, 5, 2), 'confusion', 'w-loss', 0.9),
# ('cifar10', 'mobilenetv2-115', (3, 5, 2), 'confusion', 'w-dbr', 0.9),
#
# ('cifar100', 'resnet-34', (98, 35, 11), 'confusion', 'w-aug', 0.3),
# ('cifar100', 'resnet-34', (98, 35, 11), 'confusion', 'w-bn', 0.9),
# ('cifar100', 'resnet-34', (98, 35, 11), 'confusion', 'w-loss', 0.9),
# ('cifar100', 'resnet-34', (98, 35, 11), 'confusion', 'w-dbr', 0.9),
#
#
#
# ('cifar10', 'resnet-18', (3, 5, 2), 'bias', 'w-aug', 0.7),
# ('cifar10', 'resnet-18', (3, 5, 2), 'bias', 'w-bn', 0.7),
# ('cifar10', 'resnet-18', (3, 5, 2), 'bias', 'w-bn', 0.9),
# ('cifar10', 'resnet-18', (3, 5, 2), 'bias', 'w-loss', 0.9),
# ('cifar10', 'resnet-18', (3, 5, 2), 'bias', 'w-dbr', 0.9),
#
# ('cifar10', 'vggbn-11', (3, 5, 2), 'bias', 'w-aug', 0.7),
# ('cifar10', 'vggbn-11', (3, 5, 2), 'bias', 'w-bn', 0.5),
# ('cifar10', 'vggbn-11', (3, 5, 2), 'bias', 'w-loss', 0.9),
# ('cifar10', 'vggbn-11', (3, 5, 2), 'bias', 'w-dbr', 0.9),
#
# ('cifar10', 'mobilenetv2-115', (3, 5, 2), 'bias', 'w-aug', 0.5),
# ('cifar10', 'mobilenetv2-115', (3, 5, 2), 'bias', 'w-bn', 0.5),
# ('cifar10', 'mobilenetv2-115', (3, 5, 2), 'bias', 'w-loss', 0.1),
# ('cifar10', 'mobilenetv2-115', (3, 5, 2), 'bias', 'w-dbr', 0.9),
#
# ('cifar100', 'resnet-34', (98, 35, 11), 'bias', 'w-aug', 0.7),
# ('cifar100', 'resnet-34', (98, 35, 11), 'bias', 'w-bn', 0.9),
# ('cifar100', 'resnet-34', (98, 35, 11), 'bias', 'w-loss', 0.9),
# ('cifar100', 'resnet-34', (98, 35, 11), 'bias', 'w-dbr', 0.1),
# ]

# config_list = [
# ('cifar10_multipair', 'resnet-18', (3, 5, 1, 9), 'confusion', 'w-aug', 1.0),
# ('cifar10_multipair', 'resnet-18', (3, 5, 1, 9), 'confusion', 'w-aug', 0.7),
# ('cifar10_multipair', 'resnet-18', (3, 5, 1, 9), 'confusion', 'w-bn', 0.5),
# ('cifar10_multipair', 'resnet-18', (3, 5, 1, 9), 'confusion', 'w-bn', 0.9),
# ('cifar10_multipair', 'resnet-18', (3, 5, 1, 9), 'confusion', 'w-loss', 0.9),
# ('cifar10_multipair', 'resnet-18', (3, 5, 1, 9), 'confusion', 'w-dbr', 0.9),
# ]

config_list = [
('cifar100', 'resnet-34', (98, 35, 11), 'bias', 'w-bn', 0.7),
]



def execute_cmd(dataset, model, classes, task, method, param, log_filename, t0, rep_num):
    if model == 'vggbn-11':
        model_type = 'resnet'
        model_depth = '18'
        vggbn = '_vggbn'
    else:
        model_type, model_depth = model.split('-')
        vggbn = ''

    if model == 'vggbn-11':
        model_path = 'models/cifar10_vggbn_2_4/model_best.pth.tar'
    elif model == 'resnet-18':
        model_path = 'models/cifar10_resnet18_2_4/model_best.pth.tar'
    elif model == 'mobilenetv2-115':
        model_path = 'models/cifar10_mobilenetv2/model_best.pth.tar'
    elif model == 'resnet-34':
        model_path = 'models/cifar100_resnet34/model_best.pth.tar'
    else:
        raise

    if dataset in ['cifar10_multipair']:
        dataset_param = 'cifar10'
        first, second, third, fourth = classes
    else:
        dataset_param = dataset
        first, second, third = classes

    param_name = method_properties[method]["param_name"]
    filename = method_properties[method]["filename"]
    replace = method_properties[method]["replace"]
    batch_size = method_properties[method]["batch_size"]
    extra_batch_size = method_properties[method]["extra_batch_size"]

    filepath = os.path.join('exp_7', dataset, task, 'repair_'+task+'_'+filename+vggbn+'.py')
    subfolder = dataset+'_'+task+'_'+str(first)+'_'+str(second)+'_'+str(third)+'_'+str(rep_num)
    expdir = os.path.join('runs', subfolder)
    if not os.path.isdir(expdir):
        os.mkdir(expdir)
    expname = os.path.join(subfolder, dataset+'_'+model+'_'+task+'_'+method+'_'+str(param))

    cmd = f"python3 {filepath} --net_type {model_type} --dataset {dataset_param} --depth {model_depth} --expname {expname} --epochs {epochs} --lr 0.1 --beta 1.0 --cutmix_prob 0 --pretrained {model_path} --batch_size {batch_size} --extra {extra_batch_size} --{param_name} {param}"+replace+verbose

    if dataset in ['cifar10_multipair']:
        cmd += f' --pair1a {first} --pair1b {second} --pair2a {third} --pair2b {fourth}'
    else:
        cmd += f' --first {first} --second {second} --third {third}'

    print('-'*20)
    print(rep_num, time.time()-t0, cmd)
    print('-'*20)
    with open('tmp_log.txt', 'a') as f_out:
        f_out.write(cmd+'\n')
        f_out.write(str(time.time()-t0)+'\n')
    execute(cmd)

if __name__ == '__main__':
    # ['grid', 'specific']
    mode = 'specific'
    t0 = time.time()
    if mode == 'grid':
        log_filename = 'tmp_log.txt'
        with open(log_filename, 'w') as f_out:
            pass
        for dataset, model, classes in dataset_model_classes:
            for task in tasks:
                for method in methods:
                    for param in params:
                        execute_cmd(dataset, model, classes, task, method, param, log_filename, t0, 0)
    elif mode == 'specific':
        rep_nums = 4
        log_filename = 'tmp_log_cifar_specific.txt'
        with open(log_filename, 'w') as f_out:
            pass
        for rep_num in range(rep_nums):
            for dataset, model, classes, task, method, param in config_list:
                execute_cmd(dataset, model, classes, task, method, param, log_filename, t0, rep_num)
    else:
        raise
