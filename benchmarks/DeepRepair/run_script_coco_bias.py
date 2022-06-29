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
    "filename": "exp_weighted_loss",
    "replace": "",
    # "batch_size": 64,
    # "extra_batch_size": 0,
},
'w-bn': {
    "param_name": "ratio",
    "filename": "bn",
    "replace": " --replace",
    # "batch_size": 64,
    # "extra_batch_size": 64,
},
'w-loss': {
    "param_name": "target_weight",
    "filename": "exp_weighted_loss",
    "replace": "",
    # "batch_size": 64,
    # "extra_batch_size": 0,
},
'w-dbr': {
    "param_name": "lam",
    "filename": "dbr",
    "replace": "",
    # "batch_size": 64,
    # "extra_batch_size": 6,
}
}



dataset_model_classes = [('coco', 'coco', ("bus", "person", "clock")), ('coco_gender', 'coco_gender', ("handbag", "woman", "man")), ('coco_multipair', 'coco_multipair', ("person", "bus", "mouse", "key"))]
tasks = ['confusion', 'bias']
methods = ['w-aug', 'w-bn', 'w-loss', 'w-dbr']
params = [0.1, 0.3, 0.5, 0.7, 0.9]
epochs = 18
verbose = ""

# dataset_model_classes = [('coco', 'coco', ("bus", "person", "clock")), ('coco_gender', 'coco_gender', ("handbag", "woman", "man"))]
# tasks = ['confusion']
# methods = ['w-aug', 'w-bn', 'w-loss', 'w-dbr']
# params = [0.1, 0.3, 0.5, 0.7, 0.9]
# epochs = 18

# dataset_model_classes = [('coco', 'coco', ("bus", "person", "clock")), ('coco_gender', 'coco_gender', ("skis", "woman", "man"))]
# tasks = ['bias']
# methods = ['w-aug', 'w-bn', 'w-loss', 'w-dbr']
# params = [0.1, 0.3, 0.5, 0.7, 0.9]
# epochs = 18


# config_list = [
# ('coco', 'coco', ("bus", "person", "clock"), 'bias', 'w-aug', 0.9),
# ('coco', 'coco', ("bus", "person", "clock"), 'bias', 'w-bn', 0.7),
# ('coco', 'coco', ("bus", "person", "clock"), 'bias', 'w-loss', 0.9),
# ('coco', 'coco', ("bus", "person", "clock"), 'bias', 'w-dbr', 0.1),
# ('coco_gender', 'coco_gender', ("skis", "woman", "man"), 'bias', 'w-aug', 0.9),
# ('coco_gender', 'coco_gender', ("skis", "woman", "man"), 'bias', 'w-bn', 0.9),
# ('coco_gender', 'coco_gender', ("skis", "woman", "man"), 'bias', 'w-loss', 0.9),
# ('coco_gender', 'coco_gender', ("skis", "woman", "man"), 'bias', 'w-dbr', 0.5),
# ]

config_list = [
('coco_multipair', 'coco_multipair', ("person", "bus", "mouse", "keyboard"), 'confusion', 'w-loss', 0.9),
('coco_multipair', 'coco_multipair', ("person", "bus", "mouse", "keyboard"), 'confusion', 'w-dbr', 0.9),
]

def execute_cmd(dataset, model, classes, task, method, param, log_filename, t0, rep_num):
    if model in ['coco', 'coco_multipair']:
        model_path = 'models/coco_original_model/model_best.pth.tar'
        class_num = 80
    elif model == 'coco_gender':
        model_path = 'models/cocogender_original_model/model_best.pth.tar'
        class_num = 81
    else:
        raise

    if dataset in ['coco_multipair']:
        first, second, third, fourth = classes
    else:
        first, second, third = classes

    param_name = method_properties[method]["param_name"]
    filename = method_properties[method]["filename"]
    replace = method_properties[method]["replace"]

    filepath = os.path.join('exp_9', dataset, 'confusion_and_bias', 'repair_'+task+'_'+filename+'.py')
    expdir = os.path.join('runs', dataset+'_'+task+'_'+str(first)+'_'+str(second)+'_'+str(third)+'_'+str(rep_num))
    if not os.path.isdir(expdir):
        os.mkdir(expdir)
    expname = os.path.join(expdir, dataset+'_'+task+'_'+method+'_'+str(param))

    cmd = f"python2 {filepath} --pretrained {model_path} --log_dir {expname} --ann_dir '../coco/annotations' --num_epochs {epochs} --image_dir '../coco/' --seed {rep_num*2} --class_num {class_num} --{param_name} {param}"+replace

    if dataset in ['coco_multipair']:
        cmd += f' --pair1a {first} --pair1b {second} --pair2a {third} --pair2b {fourth}'
    else:
        cmd += f' --first {first} --second {second} --third {third}'

    print('-'*20)
    print(cmd)
    print('-'*20)
    with open(log_filename, 'a') as f_out:
        f_out.write(cmd+'\n')
        f_out.write(str(time.time()-t0)+'\n')
    execute(cmd)

if __name__ == '__main__':
    # ['grid', 'specific']
    mode = 'specific'
    t0 = time.time()
    if mode == 'grid':
        log_filename = 'tmp_log_coco_bias.txt'
        with open(log_filename, 'w') as f_out:
            pass
        for dataset, model, classes in dataset_model_classes:
            for task in tasks:
                for method in methods:
                    for param in params:
                        execute_cmd(dataset, model, classes, task, method, param, log_filename, t0, 0)
    elif mode == 'specific':
        rep_nums = 4
        log_filename = 'tmp_log_coco_bias_specific.txt'
        with open(log_filename, 'w') as f_out:
            pass
        for rep_num in range(rep_nums):
            print('rep_num:', rep_num)
            for dataset, model, classes, task, method, param in config_list:
                execute_cmd(dataset, model, classes, task, method, param, log_filename, t0, rep_num)
    else:
        raise
