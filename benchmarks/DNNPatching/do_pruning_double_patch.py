import argparse
import matplotlib.pyplot as plt
from Pipeline.options import args
import random
import os
import numpy as np
from tqdm import tqdm 
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

from Architecture.double_patch_model import net, optimizer, scheduler
from DataLoaders.data import trainData, trainSubset, lf_town03_validData, lc_town03_validData, oa_town03_validData

best_failrate = 1
logfile_name = 'pruning_finetune_'+str(args.finetune_prunednet)+'_log.txt'
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
copyfile('./do_pruning_double_patch.py', args.output_dir+'do_pruning.py')
copyfile('./Architecture/double_patch_model.py', args.output_dir+'model.py')
copyfile('./DataLoaders/data.py', args.output_dir+'data.py')


def main():
    global logfile_name
    criterion = torch.nn.MSELoss()

    criterion1 = torch.nn.BCEWithLogitsLoss()
    criterion2 = torch.nn.MSELoss()

    load_state = torch.load(args.resume_from)
    net.load_state_dict(load_state)

    print('\nEvaluation only')
    if args.task == 'lf' or args.task == 'lc':
        train_loss, train_failrate = test(net, criterion, trainData)
        valid_loss, valid_failrate = test(net, criterion, trainSubset)
        lf_valid_loss, lf_valid_failrate = test(net, criterion, lf_town03_validData)
    elif args.task == 'oa':
        train_loss, train_failrate, train_obj_detectioin_frate, train_left_avoidance_frate, train_right_avoidance_frate = oa_test(net, criterion1, criterion2, trainData)
        valid_loss, valid_failrate, valid_obj_detectioin_frate, valid_left_avoidance_frate, valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, trainSubset)
        lf_valid_loss, lf_valid_failrate, lf_valid_obj_detectioin_frate, lf_valid_left_avoidance_frate, lf_valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, lf_town03_validData)
    if args.task == 'lc':
        lc_valid_loss, lc_valid_failrate = test(net, criterion, lc_town03_validData)
    if args.task == 'oa':
        oa_valid_loss, oa_valid_failrate, oa_valid_obj_detectioin_frate, oa_valid_left_avoidance_frate, oa_valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, oa_town03_validData)
    
    print('Before pruning\n')
    print('Train-Loss:  %.8f, Train Failrate:  %.3f\n' % (train_loss, train_failrate))
    if args.task == 'oa':
        print('Train obj detectioin frate:  %.3f\n' % (train_obj_detectioin_frate))
        print('Train left avoidance frate:  %.3f\n' % (train_left_avoidance_frate))
        print('Train right avoidance frate:  %.3f\n' % (train_right_avoidance_frate))
    print('Valid-Loss: %.8f, Valid Failrate:  %.3f\n' % (valid_loss, valid_failrate))
    if args.task == 'oa':
        print('Valid obj detectioin frate:  %.3f\n' % (valid_obj_detectioin_frate))
        print('Valid left avoidance frate:  %.3f\n' % (valid_left_avoidance_frate))
        print('Valid right avoidance frate:  %.3f\n' % (valid_right_avoidance_frate))
    print('LF Valid-Loss: %.8f, LF Valid Failrate:  %.3f\n' % (lf_valid_loss, lf_valid_failrate))
    if args.task == 'oa':
        print('LF Valid obj detectioin frate:  %.3f\n' % (lf_valid_obj_detectioin_frate))
        print('LF Valid left avoidance frate:  %.3f\n' % (lf_valid_left_avoidance_frate))
        print('LF Valid right avoidance frate:  %.3f\n' % (lf_valid_right_avoidance_frate))
    if args.task == 'lc':
        print('LC Valid-Loss: %.8f, LC Valid Failrate:  %.3f\n' % (lc_valid_loss, lc_valid_failrate))
    if args.task == 'oa':
        print('OA Valid-Loss: %.8f, OA Valid Failrate:  %.3f\n' % (oa_valid_loss, oa_valid_failrate))
        print('OA Valid obj detectioin frate:  %.3f\n' % (oa_valid_obj_detectioin_frate))
        print('OA Valid left avoidance frate:  %.3f\n' % (oa_valid_left_avoidance_frate))
        print('OAValid right avoidance frate:  %.3f\n' % (oa_valid_right_avoidance_frate))

    with open(os.path.join(args.output_dir, logfile_name), 'w') as f:
        f.write('Before pruning')
        f.write('Train-Loss:  %.8f, Train Failrate:  %.3f\n' % (train_loss, train_failrate))
        if args.task == 'oa':
            f.write('Train obj detectioin frate:  %.3f\n' % (train_obj_detectioin_frate))
            f.write('Train left avoidance frate:  %.3f\n' % (train_left_avoidance_frate))
            f.write('Train right avoidance frate:  %.3f\n' % (train_right_avoidance_frate))
        f.write('Valid-Loss: %.8f, Valid Failrate:  %.3f\n' % (valid_loss, valid_failrate))
        if args.task == 'oa':
            f.write('Valid obj detectioin frate:  %.3f\n' % (valid_obj_detectioin_frate))
            f.write('Valid left avoidance frate:  %.3f\n' % (valid_left_avoidance_frate))
            f.write('Valid right avoidance frate:  %.3f\n' % (valid_right_avoidance_frate))
        f.write('LF Valid-Loss: %.8f, LF Valid Failrate:  %.3f\n' % (lf_valid_loss, lf_valid_failrate))
        if args.task == 'oa':
            f.write('LF Valid obj detectioin frate:  %.3f\n' % (lf_valid_obj_detectioin_frate))
            f.write('LF Valid left avoidance frate:  %.3f\n' % (lf_valid_left_avoidance_frate))
            f.write('LF Valid right avoidance frate:  %.3f\n' % (lf_valid_right_avoidance_frate))
        if args.task == 'lc':
            f.write('LC Valid-Loss: %.8f, LC Valid Failrate:  %.3f\n' % (lc_valid_loss, lc_valid_failrate))
        if args.task == 'oa':
            f.write('OA Valid-Loss: %.8f, OA Valid Failrate:  %.3f\n' % (oa_valid_loss, oa_valid_failrate))
            f.write('OA Valid obj detectioin frate:  %.3f\n' % (oa_valid_obj_detectioin_frate))
            f.write('OA Valid left avoidance frate:  %.3f\n' % (oa_valid_left_avoidance_frate))
            f.write('OAValid right avoidance frate:  %.3f\n' % (oa_valid_right_avoidance_frate))

    # total pruning
    total = 0
    for name, m in net.named_modules():
        if args.train_patch:
            if 'datch' in name:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    total += m.weight.data.numel()
        else:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                total += m.weight.data.numel()
    all_weights = torch.zeros(total)
    index = 0
    for name, m in net.named_modules():
        if args.train_patch:
            if 'datch' in name:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    size = m.weight.data.numel()
                    all_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
                    index += size
        else:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                size = m.weight.data.numel()
                all_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
                index += size

    #plot_weights_histogram(all_weights)

    y, i = torch.sort(all_weights)
    thre_index = int(total * args.pruning_percent)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    with open(os.path.join(args.output_dir, logfile_name), 'a+') as f:
        f.write('Pruning threshold: {}\n'.format(thre))
    zero_flag = False
    for k, (name, m) in enumerate(net.named_modules()):
        if args.train_patch:
            if 'datch' in name:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(thre).float().cuda()
                    pruned = pruned + mask.numel() - torch.sum(mask)
                    m.weight.data.mul_(mask)
                    if int(torch.sum(mask)) == 0:
                        zero_flag = True
                    print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                        format(k, mask.numel(), int(torch.sum(mask))))
                    with open(os.path.join(args.output_dir, logfile_name), 'a+') as f:
                        f.write('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.format(k, mask.numel(), int(torch.sum(mask))))
        else:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                pruned = pruned + mask.numel() - torch.sum(mask)
                m.weight.data.mul_(mask)
                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.format(k, mask.numel(), int(torch.sum(mask))))
                with open(os.path.join(args.output_dir, logfile_name), 'a+') as f:
                    f.write('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.format(k, mask.numel(), int(torch.sum(mask))))

    # TODO fix counting number of pruned parameters
    print('Total params: {}, Pruned params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    if zero_flag:
        print("There exists a layer with 0 parameters left.")
    with open(os.path.join(args.output_dir, logfile_name), 'a+') as f:
        f.write('Total params: {}, Pruned params: {}, Pruned ratio: {}\n'.format(total, pruned, pruned/total))
        if zero_flag:
            f.write("There exists a layer with 0 parameters left.\n")

    print('After Pruning\n')
    if args.task == 'lf' or args.task == 'lc':
        train_loss, train_failrate = test(net, criterion, trainData)
        valid_loss, valid_failrate = test(net, criterion, trainSubset)
        lf_valid_loss, lf_valid_failrate = test(net, criterion, lf_town03_validData)
    elif args.task == 'oa':
        train_loss, train_failrate, train_obj_detectioin_frate, train_left_avoidance_frate, train_right_avoidance_frate = oa_test(net, criterion1, criterion2, trainData)
        valid_loss, valid_failrate, valid_obj_detectioin_frate, valid_left_avoidance_frate, valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, trainSubset)
        lf_valid_loss, lf_valid_failrate, lf_valid_obj_detectioin_frate, lf_valid_left_avoidance_frate, lf_valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, lf_town03_validData)
    if args.task == 'lc':
        lc_valid_loss, lc_valid_failrate = test(net, criterion, lc_town03_validData)
    if args.task == 'oa':
        oa_valid_loss, oa_valid_failrate, oa_valid_obj_detectioin_frate, oa_valid_left_avoidance_frate, oa_valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, oa_town03_validData)

    print('Train-Loss:  %.8f, Train Failrate:  %.3f\n' % (train_loss, train_failrate))
    if args.task == 'oa':
        print('Train obj detectioin frate:  %.3f\n' % (train_obj_detectioin_frate))
        print('Train left avoidance frate:  %.3f\n' % (train_left_avoidance_frate))
        print('Train right avoidance frate:  %.3f\n' % (train_right_avoidance_frate))
    print('Valid-Loss: %.8f, Valid Failrate:  %.3f\n' % (valid_loss, valid_failrate))
    if args.task == 'oa':
        print('Valid obj detectioin frate:  %.3f\n' % (valid_obj_detectioin_frate))
        print('Valid left avoidance frate:  %.3f\n' % (valid_left_avoidance_frate))
        print('Valid right avoidance frate:  %.3f\n' % (valid_right_avoidance_frate))
    print('LF Valid-Loss: %.8f, LF Valid Failrate:  %.3f\n' % (lf_valid_loss, lf_valid_failrate))
    if args.task == 'oa':
        print('LF Valid obj detectioin frate:  %.3f\n' % (lf_valid_obj_detectioin_frate))
        print('LF Valid left avoidance frate:  %.3f\n' % (lf_valid_left_avoidance_frate))
        print('LF Valid right avoidance frate:  %.3f\n' % (lf_valid_right_avoidance_frate))
    if args.task == 'lc':
        print('LC Valid-Loss: %.8f, LC Valid Failrate:  %.3f\n' % (lc_valid_loss, lc_valid_failrate))
    if args.task == 'oa':
        print('OA Valid-Loss: %.8f, OA Valid Failrate:  %.3f\n' % (oa_valid_loss, oa_valid_failrate))
        print('OA Valid obj detectioin frate:  %.3f\n' % (oa_valid_obj_detectioin_frate))
        print('OA Valid left avoidance frate:  %.3f\n' % (oa_valid_left_avoidance_frate))
        print('OA Valid right avoidance frate:  %.3f\n' % (oa_valid_right_avoidance_frate))

    with open(os.path.join(args.output_dir, logfile_name), 'a+') as f:
        f.write('After Pruning\n')
        f.write('Train-Loss:  %.8f, Train Failrate:  %.3f\n' % (train_loss, train_failrate))
        if args.task == 'oa':
            f.write('Train obj detectioin frate:  %.3f\n' % (train_obj_detectioin_frate))
            f.write('Train left avoidance frate:  %.3f\n' % (train_left_avoidance_frate))
            f.write('Train right avoidance frate:  %.3f\n' % (train_right_avoidance_frate))
        f.write('Valid-Loss: %.8f, Valid Failrate:  %.3f\n' % (valid_loss, valid_failrate))
        if args.task == 'oa':
            f.write('Valid obj detectioin frate:  %.3f\n' % (valid_obj_detectioin_frate))
            f.write('Valid left avoidance frate:  %.3f\n' % (valid_left_avoidance_frate))
            f.write('Valid right avoidance frate:  %.3f\n' % (valid_right_avoidance_frate))
        f.write('LF Valid-Loss: %.8f, LF Valid Failrate:  %.3f\n' % (lf_valid_loss, lf_valid_failrate))
        if args.task == 'oa':
            f.write('LF Valid obj detectioin frate:  %.3f\n' % (lf_valid_obj_detectioin_frate))
            f.write('LF Valid left avoidance frate:  %.3f\n' % (lf_valid_left_avoidance_frate))
            f.write('LF Valid right avoidance frate:  %.3f\n' % (lf_valid_right_avoidance_frate))
        if args.task == 'lc':
            f.write('LC Valid-Loss: %.8f, LC Valid Failrate:  %.3f\n' % (lc_valid_loss, lc_valid_failrate))
        if args.task == 'oa':
            f.write('OA Valid-Loss: %.8f, OA Valid Failrate:  %.3f\n' % (oa_valid_loss, oa_valid_failrate))
            f.write('OA Valid obj detectioin frate:  %.3f\n' % (oa_valid_obj_detectioin_frate))
            f.write('OA Valid left avoidance frate:  %.3f\n' % (oa_valid_left_avoidance_frate))
            f.write('OAValid right avoidance frate:  %.3f\n' % (oa_valid_right_avoidance_frate))

    if args.finetune_prunednet:
        global best_failrate
        lossTr = torch.zeros(args.finetune_epochs, 1)
        lossVa = torch.zeros(args.finetune_epochs, 1)
        errTr = torch.zeros(args.finetune_epochs, 1)
        errVa = torch.zeros(args.finetune_epochs, 1)
        lf_town03_lossVa = torch.zeros(args.finetune_epochs, 1)
        lf_town03_errVa = torch.zeros(args.finetune_epochs, 1)
        if args.task == 'oa':
            errTr_od = torch.zeros(args.finetune_epochs, 1)
            errTr_la = torch.zeros(args.finetune_epochs, 1)
            errTr_ra = torch.zeros(args.finetune_epochs, 1)
            errVa_od = torch.zeros(args.finetune_epochs, 1)
            errVa_la = torch.zeros(args.finetune_epochs, 1)
            errVa_ra = torch.zeros(args.finetune_epochs, 1)
            lf_town03_errVa_od = torch.zeros(args.finetune_epochs, 1)
            lf_town03_errVa_la = torch.zeros(args.finetune_epochs, 1)
            lf_town03_errVa_ra = torch.zeros(args.finetune_epochs, 1)
        if args.task == 'lc':
            lc_town03_lossVa = torch.zeros(args.finetune_epochs, 1)
            lc_town03_errVa = torch.zeros(args.finetune_epochs, 1)
        if args.task == 'oa':
            oa_town03_lossVa = torch.zeros(args.finetune_epochs, 1)
            oa_town03_errVa = torch.zeros(args.finetune_epochs, 1)
            oa_town03_errVa_od = torch.zeros(args.finetune_epochs, 1)
            oa_town03_errVa_la = torch.zeros(args.finetune_epochs, 1)
            oa_town03_errVa_ra = torch.zeros(args.finetune_epochs, 1)

        with open(os.path.join(args.output_dir, logfile_name), 'a+') as f:
            f.write('Fine Tuning\n')
        is_best = 1
        for epoch in range(args.finetune_epochs):
            print('\nEpoch: [%d | %d]' % (epoch + 1, args.finetune_epochs))
            num_parameters = get_zero_param(net)
            print('Zero parameters: {}'.format(num_parameters))
            num_parameters = sum([param.nelement() for param in net.parameters()])
            print('Parameters: {}'.format(num_parameters))

            if args.task == 'lf' or args.task == 'lc':
                train_loss, train_failrate = train(net, criterion, optimizer, epoch, trainData)
                valid_loss, valid_failrate = test(net, criterion, trainSubset)
                lf_valid_loss, lf_valid_failrate = test(net, criterion, lf_town03_validData)
            elif args.task == 'oa':
                train_loss, train_failrate, train_obj_detectioin_frate, train_left_avoidance_frate, train_right_avoidance_frate = oa_train(net, criterion1, criterion2, optimizer, epoch, trainData)
                valid_loss, valid_failrate, valid_obj_detectioin_frate, valid_left_avoidance_frate, valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, trainSubset)
                lf_valid_loss, lf_valid_failrate, lf_valid_obj_detectioin_frate, lf_valid_left_avoidance_frate, lf_valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, lf_town03_validData)
            if args.task == 'lc':
                lc_valid_loss, lc_valid_failrate = test(net, criterion, lc_town03_validData)
            if args.task == 'oa':
                oa_valid_loss, oa_valid_failrate, oa_valid_obj_detectioin_frate, oa_valid_left_avoidance_frate, oa_valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, oa_town03_validData)


            lossTr.select(0, epoch).copy_(train_loss)
            lossVa.select(0, epoch).copy_(valid_loss)
            errTr.select(0, epoch).fill_(train_failrate)
            errVa.select(0, epoch).fill_(valid_failrate)
            lf_town03_lossVa.select(0, epoch).copy_(lf_valid_loss)
            lf_town03_errVa.select(0, epoch).fill_(lf_valid_failrate)
            if args.task == 'oa':
                errTr_od.select(0, epoch).fill_(train_obj_detectioin_frate)
                errTr_la.select(0, epoch).fill_(train_left_avoidance_frate)
                errTr_ra.select(0, epoch).fill_(train_right_avoidance_frate)
                errVa_od.select(0, epoch).fill_(valid_obj_detectioin_frate)
                errVa_la.select(0, epoch).fill_(valid_left_avoidance_frate)
                errVa_ra.select(0, epoch).fill_(valid_right_avoidance_frate)
                lf_town03_errVa_od.select(0, epoch).fill_(lf_valid_obj_detectioin_frate)
                lf_town03_errVa_la.select(0, epoch).fill_(lf_valid_left_avoidance_frate)
                lf_town03_errVa_ra.select(0, epoch).fill_(lf_valid_right_avoidance_frate)

            if args.task == 'lc':
                lc_town03_lossVa.select(0, epoch).copy_(lc_valid_loss)
                lc_town03_errVa.select(0, epoch).fill_(lc_valid_failrate)

            if args.task == 'oa':
                lf_town03_errVa_od.select(0, epoch).fill_(oa_valid_obj_detectioin_frate)
                lf_town03_errVa_la.select(0, epoch).fill_(oa_valid_left_avoidance_frate)
                lf_town03_errVa_ra.select(0, epoch).fill_(oa_valid_right_avoidance_frate)


            np.save(args.output_dir+'trLoss.npy', lossTr.numpy())
            np.save(args.output_dir+'trErr.npy', errTr.numpy())
            np.save(args.output_dir+'vaLoss.npy', lossVa.numpy())
            np.save(args.output_dir+'vaErr.npy', errVa.numpy())
            np.save(args.output_dir+'lf_town03_vaLoss.npy', lf_town03_lossVa.numpy())
            np.save(args.output_dir+'lf_town03_vaErr.npy', lf_town03_errVa.numpy())
            if args.task == 'oa':
                np.save(args.output_dir+'errTr_od.npy', errTr_od.numpy())
                np.save(args.output_dir+'errTr_la.npy', errTr_la.numpy())
                np.save(args.output_dir+'errTr_ra.npy', errTr_ra.numpy())
                np.save(args.output_dir+'errVa_od.npy', errVa_od.numpy())
                np.save(args.output_dir+'errVa_la.npy', errVa_la.numpy())
                np.save(args.output_dir+'errVa_ra.npy', errVa_ra.numpy())
                np.save(args.output_dir+'lf_town03_errVa_od.npy', lf_town03_errVa_od.numpy())
                np.save(args.output_dir+'lf_town03_errVa_la.npy', lf_town03_errVa_la.numpy())
                np.save(args.output_dir+'lf_town03_errVa_ra.npy', lf_town03_errVa_ra.numpy())
            if args.task == 'lc':
                np.save(args.output_dir+'lc_town03_vaLoss.npy', lc_town03_lossVa.numpy())
                np.save(args.output_dir+'lc_town03_vaErr.npy', lc_town03_errVa.numpy())
            if args.task == 'oa':
                np.save(args.output_dir+'oa_town03_errVa_od.npy', oa_town03_errVa_od.numpy())
                np.save(args.output_dir+'oa_town03_errVa_la.npy', oa_town03_errVa_la.numpy())
                np.save(args.output_dir+'oa_town03_errVa_ra.npy', oa_town03_errVa_ra.numpy())

            print('Train-Loss:  %.8f, Train Failrate:  %.3f\n' % (train_loss, train_failrate))
            if args.task == 'oa':
                print('Train obj detectioin frate:  %.3f\n' % (train_obj_detectioin_frate))
                print('Train left avoidance frate:  %.3f\n' % (train_left_avoidance_frate))
                print('Train right avoidance frate:  %.3f\n' % (train_right_avoidance_frate))
            print('Valid-Loss: %.8f, Valid Failrate:  %.3f\n' % (valid_loss, valid_failrate))
            if args.task == 'oa':
                print('Valid obj detectioin frate:  %.3f\n' % (valid_obj_detectioin_frate))
                print('Valid left avoidance frate:  %.3f\n' % (valid_left_avoidance_frate))
                print('Valid right avoidance frate:  %.3f\n' % (valid_right_avoidance_frate))
            print('LF Valid-Loss: %.8f, LF Valid Failrate:  %.3f\n' % (lf_valid_loss, lf_valid_failrate))
            if args.task == 'oa':
                print('LF Valid obj detectioin frate:  %.3f\n' % (lf_valid_obj_detectioin_frate))
                print('LF Valid left avoidance frate:  %.3f\n' % (lf_valid_left_avoidance_frate))
                print('LF Valid right avoidance frate:  %.3f\n' % (lf_valid_right_avoidance_frate))
            if args.task == 'lc':
                print('LC Valid-Loss: %.8f, LC Valid Failrate:  %.3f\n' % (lc_valid_loss, lc_valid_failrate))
            if args.task == 'oa':
                print('OA Valid-Loss: %.8f, OA Valid Failrate:  %.3f\n' % (oa_valid_loss, oa_valid_failrate))
                print('OA Valid obj detectioin frate:  %.3f\n' % (oa_valid_obj_detectioin_frate))
                print('OA Valid left avoidance frate:  %.3f\n' % (oa_valid_left_avoidance_frate))
                print('OA Valid right avoidance frate:  %.3f\n' % (oa_valid_right_avoidance_frate))

            with open(os.path.join(args.output_dir, logfile_name), 'a+') as f:
                f.write('After Pruning\n')
                f.write('Train-Loss:  %.8f, Train Failrate:  %.3f\n' % (train_loss, train_failrate))
                if args.task == 'oa':
                    f.write('Train obj detectioin frate:  %.3f\n' % (train_obj_detectioin_frate))
                    f.write('Train left avoidance frate:  %.3f\n' % (train_left_avoidance_frate))
                    f.write('Train right avoidance frate:  %.3f\n' % (train_right_avoidance_frate))
                f.write('Valid-Loss: %.8f, Valid Failrate:  %.3f\n' % (valid_loss, valid_failrate))
                if args.task == 'oa':
                    f.write('Valid obj detectioin frate:  %.3f\n' % (valid_obj_detectioin_frate))
                    f.write('Valid left avoidance frate:  %.3f\n' % (valid_left_avoidance_frate))
                    f.write('Valid right avoidance frate:  %.3f\n' % (valid_right_avoidance_frate))
                f.write('LF Valid-Loss: %.8f, LF Valid Failrate:  %.3f\n' % (lf_valid_loss, lf_valid_failrate))
                if args.task == 'oa':
                    f.write('LF Valid obj detectioin frate:  %.3f\n' % (lf_valid_obj_detectioin_frate))
                    f.write('LF Valid left avoidance frate:  %.3f\n' % (lf_valid_left_avoidance_frate))
                    f.write('LF Valid right avoidance frate:  %.3f\n' % (lf_valid_right_avoidance_frate))
                if args.task == 'lc':
                    f.write('LC Valid-Loss: %.8f, LC Valid Failrate:  %.3f\n' % (lc_valid_loss, lc_valid_failrate))
                if args.task == 'oa':
                    f.write('OA Valid-Loss: %.8f, OA Valid Failrate:  %.3f\n' % (oa_valid_loss, oa_valid_failrate))
                    f.write('OA Valid obj detectioin frate:  %.3f\n' % (oa_valid_obj_detectioin_frate))
                    f.write('OA Valid left avoidance frate:  %.3f\n' % (oa_valid_left_avoidance_frate))
                    f.write('OA Valid right avoidance frate:  %.3f\n' % (oa_valid_right_avoidance_frate))

            # save model        
            is_best = valid_failrate < best_failrate
            best_failrate = min(valid_failrate, best_failrate)
            
            if is_best:
                print('Is best model!')
                with open(os.path.join(args.output_dir, logfile_name), 'a+') as f:
                    f.write('Is best model!\n')
                torch.save(net.state_dict(), args.output_dir+'pruned_bestmodel_epoch_'+str(epoch)+'.pth')


        if args.task == 'oa':
            plt.figure()
            plt.plot(errTr_od.numpy(), label='Err Tr od')
            plt.plot(errVa_od.numpy(), label='Err Va od')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'tr_err_od.png')
            plt.close()

            plt.figure()
            plt.plot(errTr_la.numpy(), label='Err Tr la')
            plt.plot(errVa_la.numpy(), label='Err Va la')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'tr_err_la.png')
            plt.close()

            plt.figure()
            plt.plot(errTr_ra.numpy(), label='Err Tr ra')
            plt.plot(errVa_ra.numpy(), label='Err Va ra')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'tr_err_ra.png')
            plt.close()

            plt.figure()
            plt.plot(lf_town03_errVa_od.numpy(), label='LF Err Va od')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'lf_town03_va_err_od.png')
            plt.close()

            plt.figure()
            plt.plot(lf_town03_errVa_la.numpy(), label='LF Err Va la')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'lf_town03_va_err_la.png')
            plt.close()

            plt.figure()
            plt.plot(lf_town03_errVa_ra.numpy(), label='LF Err Va ra')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'lf_town03_va_err_ra.png')
            plt.close()


            plt.figure()
            plt.plot(oa_town03_lossVa.numpy(), label='OA loss Va')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('loss', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'oa_town03_va_loss_curve.png')
            plt.close()

            plt.figure()
            plt.plot(oa_town03_errVa.numpy(), label='OA Err Va')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'oa_town03_va_err_str.png')
            plt.close()

            plt.figure()
            plt.plot(oa_town03_errVa_od.numpy(), label='OA Err Va od')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'oa_town03_va_err_od.png')
            plt.close()

            plt.figure()
            plt.plot(oa_town03_errVa_la.numpy(), label='OA Err Va la')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'oa_town03_va_err_la.png')
            plt.close()

            plt.figure()
            plt.plot(oa_town03_errVa_ra.numpy(), label='OA Err Va ra')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'oa_town03_va_err_ra.png')
            plt.close()

            

        plt.figure()
        plt.plot(lossTr.numpy(), label='Tr loss')
        plt.plot(lossVa.numpy(), label='Va loss')
        plt.legend(loc='upper center', fontsize=22)
        plt.xlabel('Epoch number', fontsize=22)
        plt.ylabel('loss', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(args.output_dir+'tr_loss_curve.png')
        plt.close()

        plt.figure()
        plt.plot(errTr.numpy(), label='Tr err-rate')
        plt.plot(errVa.numpy(), label='Va err-rate')
        plt.legend(loc='upper center', fontsize=22)
        plt.xlabel('Epoch number', fontsize=22)
        plt.ylabel('error rate', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(args.output_dir+'tr_error_curve.png')
        plt.close()

        plt.figure()
        plt.plot(lf_town03_lossVa.numpy(), label='Va loss')
        plt.legend(loc='upper center', fontsize=22)
        plt.xlabel('Epoch number', fontsize=22)
        plt.ylabel('loss', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(args.output_dir+'lf_town03_va_loss_curve.png')
        plt.close()

        plt.figure()
        plt.plot(lf_town03_errVa.numpy(), label='Va err-rate')
        plt.legend(loc='upper center', fontsize=22)
        plt.xlabel('Epoch number', fontsize=22)
        plt.ylabel('error rate', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(args.output_dir+'lf_town03_va_error_curve.png')
        plt.close()

        if args.task == 'lc':
            plt.figure()
            plt.plot(lc_town03_lossVa.numpy(), label='Va loss')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('loss', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'lc_town03_va_loss_curve.png')
            plt.close()

            plt.figure()
            plt.plot(lc_town03_errVa.numpy(), label='Va err-rate')
            plt.legend(loc='upper center', fontsize=22)
            plt.xlabel('Epoch number', fontsize=22)
            plt.ylabel('error rate', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(args.output_dir+'lc_town03_va_error_curve.png')
            plt.close()

        print('After Finetuning best model results')

        if args.task == 'lf' or args.task == 'lc':
            train_loss, train_failrate = test(net, criterion, trainData)
            valid_loss, valid_failrate = test(net, criterion, trainSubset)
            lf_valid_loss, lf_valid_failrate = test(net, criterion, lf_town03_validData)
        elif args.task == 'oa':
            train_loss, train_failrate, train_obj_detectioin_frate, train_left_avoidance_frate, train_right_avoidance_frate = oa_test(net, criterion1, criterion2, trainData)
            valid_loss, valid_failrate, valid_obj_detectioin_frate, valid_left_avoidance_frate, valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, trainSubset)
            lf_valid_loss, lf_valid_failrate, lf_valid_obj_detectioin_frate, lf_valid_left_avoidance_frate, lf_valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, lf_town03_validData)
        if args.task == 'lc':
            lc_valid_loss, lc_valid_failrate = test(net, criterion, lc_town03_validData)
        if args.task == 'oa':
            oa_valid_loss, oa_valid_failrate, oa_valid_obj_detectioin_frate, oa_valid_left_avoidance_frate, oa_valid_right_avoidance_frate = oa_test(net, criterion1, criterion2, oa_town03_validData)

        print('Train-Loss:  %.8f, Train Failrate:  %.3f\n' % (train_loss, train_failrate))
        if args.task == 'oa':
            print('Train obj detectioin frate:  %.3f\n' % (train_obj_detectioin_frate))
            print('Train left avoidance frate:  %.3f\n' % (train_left_avoidance_frate))
            print('Train right avoidance frate:  %.3f\n' % (train_right_avoidance_frate))
        print('Valid-Loss: %.8f, Valid Failrate:  %.3f\n' % (valid_loss, valid_failrate))
        if args.task == 'oa':
            print('Valid obj detectioin frate:  %.3f\n' % (valid_obj_detectioin_frate))
            print('Valid left avoidance frate:  %.3f\n' % (valid_left_avoidance_frate))
            print('Valid right avoidance frate:  %.3f\n' % (valid_right_avoidance_frate))
        print('LF Valid-Loss: %.8f, LF Valid Failrate:  %.3f\n' % (lf_valid_loss, lf_valid_failrate))
        if args.task == 'oa':
            print('LF Valid obj detectioin frate:  %.3f\n' % (lf_valid_obj_detectioin_frate))
            print('LF Valid left avoidance frate:  %.3f\n' % (lf_valid_left_avoidance_frate))
            print('LF Valid right avoidance frate:  %.3f\n' % (lf_valid_right_avoidance_frate))
        if args.task == 'lc':
            print('LC Valid-Loss: %.8f, LC Valid Failrate:  %.3f\n' % (lc_valid_loss, lc_valid_failrate))
        if args.task == 'oa':
            print('OA Valid-Loss: %.8f, OA Valid Failrate:  %.3f\n' % (oa_valid_loss, oa_valid_failrate))
            print('OA Valid obj detectioin frate:  %.3f\n' % (oa_valid_obj_detectioin_frate))
            print('OA Valid left avoidance frate:  %.3f\n' % (oa_valid_left_avoidance_frate))
            print('OA Valid right avoidance frate:  %.3f\n' % (oa_valid_right_avoidance_frate))



        with open(os.path.join(args.output_dir, logfile_name), 'a+') as f:
            f.write('After Pruning\n')
            f.write('Train-Loss:  %.8f, Train Failrate:  %.3f\n' % (train_loss, train_failrate))
            if args.task == 'oa':
                f.write('Train obj detectioin frate:  %.3f\n' % (train_obj_detectioin_frate))
                f.write('Train left avoidance frate:  %.3f\n' % (train_left_avoidance_frate))
                f.write('Train right avoidance frate:  %.3f\n' % (train_right_avoidance_frate))
            f.write('Valid-Loss: %.8f, Valid Failrate:  %.3f\n' % (valid_loss, valid_failrate))
            if args.task == 'oa':
                f.write('Valid obj detectioin frate:  %.3f\n' % (valid_obj_detectioin_frate))
                f.write('Valid left avoidance frate:  %.3f\n' % (valid_left_avoidance_frate))
                f.write('Valid right avoidance frate:  %.3f\n' % (valid_right_avoidance_frate))
            f.write('LF Valid-Loss: %.8f, LF Valid Failrate:  %.3f\n' % (lf_valid_loss, lf_valid_failrate))
            if args.task == 'oa':
                f.write('LF Valid obj detectioin frate:  %.3f\n' % (lf_valid_obj_detectioin_frate))
                f.write('LF Valid left avoidance frate:  %.3f\n' % (lf_valid_left_avoidance_frate))
                f.write('LF Valid right avoidance frate:  %.3f\n' % (lf_valid_right_avoidance_frate))
            if args.task == 'lc':
                f.write('LC Valid-Loss: %.8f, LC Valid Failrate:  %.3f\n' % (lc_valid_loss, lc_valid_failrate))
            if args.task == 'oa':
                f.write('OA Valid-Loss: %.8f, OA Valid Failrate:  %.3f\n' % (oa_valid_loss, oa_valid_failrate))
                f.write('OA Valid obj detectioin frate:  %.3f\n' % (oa_valid_obj_detectioin_frate))
                f.write('OA Valid left avoidance frate:  %.3f\n' % (oa_valid_left_avoidance_frate))
                f.write('OAValid right avoidance frate:  %.3f\n' % (oa_valid_right_avoidance_frate))



def plot_weights_histogram(weights):
    plt.figure()
    hist1 = plt.hist(weights.numpy(), bins=100)
    plt.title('conv weights histogram')
    plt.xlabel("weights magnitude")
    plt.ylabel("Frequency")
    plt.savefig('all_weights_histogram.png')
    plt.close()

def get_zero_param(net):
    total = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            total += torch.sum(m.weight.data.eq(0))
    return total
    

def train(net, criterion, optimizer, epoch, data):
    it = iter(data)
    num_batches = len(data)
    train_loss = 0
    failures = 0
    net.train()
    for bnum in tqdm(range(num_batches)):
        batch_data = next(it)
        imgs = batch_data['image'].cuda()
        status_inputs = batch_data['status_input'].type('torch.FloatTensor').cuda()
        labels = batch_data['label'].type('torch.FloatTensor').cuda()
        imgsV = Variable(imgs)
        status_inputsV = Variable(status_inputs)
        labelsV = Variable(labels)
        inputsV = [imgsV, status_inputsV]
        output = net(inputsV)
        if args.train_patch:
            output = output[:, net.labels_count:]
        loss = criterion(output, labelsV)
        train_loss += loss.data
        failures += torch.sum(torch.abs(output.data[:, -1] - labels[:, -1]) > args.threshold)
        optimizer.zero_grad()
        loss.backward()
        #-----------------------------------------
        for name, m in net.named_modules():
            # print(k, m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if args.train_patch:
                    if 'patch' in name:
                        weight_copy = m.weight.data.abs().clone()
                        mask = weight_copy.gt(0).float().cuda()
                        m.weight.grad.data.mul_(mask)
                else:
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().cuda()
                    m.weight.grad.data.mul_(mask)
        #-----------------------------------------
        optimizer.step()
    train_loss /= num_batches
    train_failrate = float(failures)/float(num_batches * args.batch_size)
    return train_loss, train_failrate

def oa_train(net, criterion1, criterion2, optimizer, epoch, data):
    it = iter(data)
    num_batches = len(data)
    train_loss = 0
    failures = 0
    total_object_detection_correct = 0
    total_left_avoidance_correct = 0
    total_right_avoidance_correct = 0

    net.train()
    for bnum in tqdm(range(num_batches)):
        batch_data = next(it)
        imgs = batch_data['image'].cuda()
        status_inputs = batch_data['status_input'].type('torch.FloatTensor').cuda()
        labels = batch_data['label'].type('torch.FloatTensor').cuda()
        imgsV = Variable(imgs)
        status_inputsV = Variable(status_inputs)
        labels1 = labels[:, :net.aux_labels_count]
        labels2 = labels[:, net.aux_labels_count:]
        labels1V = Variable(labels1)
        labels2V = Variable(labels2)
        inputsV = [imgsV, status_inputsV]
        output = net(inputsV)
        if args.train_patch:
            output = output[:, 2*net.labels_count:]
        output1 = output[:, :net.aux_labels_count]
        output1_sigmoid = torch.sigmoid(output1)
        output2 = output[:, net.aux_labels_count:]
        loss1 = criterion1(output1, labels1V)
        loss2 = criterion2(output2, labels2V)
        loss = loss1 + loss2

        train_loss += loss.data
        failures += torch.sum(torch.abs(output.data[:, -1] - labels[:, -1]) > args.threshold)
        optimizer.zero_grad()
        loss.backward()
        #-----------------------------------------
        for name, m in net.named_modules():
            # print(k, m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if args.train_patch:
                    if 'datch' in name:
                        weight_copy = m.weight.data.abs().clone()
                        mask = weight_copy.gt(0).float().cuda()
                        m.weight.grad.data.mul_(mask)
                else:
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().cuda()
                    m.weight.grad.data.mul_(mask)
        #-----------------------------------------
        optimizer.step()
        
        output_decision = (output1_sigmoid.data[:, 0]>0.5).float()
        out_decision = (output_decision == labels[:, 0]).float()
        correct = out_decision.float().sum()
        total_object_detection_correct += correct

        output_decision = (output1_sigmoid.data[:, 1]>0.5).float()
        out_decision = (output_decision == labels[:, 1]).float()
        correct = out_decision.float().sum()
        total_left_avoidance_correct += correct

        output_decision = (output1_sigmoid.data[:, 2]>0.5).float()
        out_decision = (output_decision == labels[:, 2]).float()
        correct = out_decision.float().sum()
        total_right_avoidance_correct += correct

    train_loss /= num_batches
    train_failrate = float(failures)/float(num_batches * args.batch_size)
    total_object_detection_correct = float(total_object_detection_correct)/float(num_batches * args.batch_size)
    total_left_avoidance_correct = float(total_left_avoidance_correct)/float(num_batches * args.batch_size)
    total_right_avoidance_correct = float(total_right_avoidance_correct)/float(num_batches * args.batch_size)
    object_detection_failrate = 1.0 - total_object_detection_correct
    left_avoidance_failrate = 1.0 - total_left_avoidance_correct
    right_avoidance_failrate = 1.0 - total_right_avoidance_correct
    return train_loss, train_failrate, object_detection_failrate, left_avoidance_failrate, right_avoidance_failrate


def test(net, criterion, data):
    it = iter(data)
    num_batches = len(data)
    test_loss = 0
    failures = 0
    net.eval()
    for bnum in tqdm(range(num_batches)):
        batch_data = next(it)
        imgs = batch_data['image'].cuda()
        status_inputs = batch_data['status_input'].type('torch.FloatTensor').cuda()
        labels = batch_data['label'].type('torch.FloatTensor').cuda()
        imgsV = Variable(imgs)
        status_inputsV = Variable(status_inputs)
        labelsV = Variable(labels)
        inputsV = [imgsV, status_inputsV]
        output = net(inputsV)
        if args.train_patch:
            output = output[:, net.labels_count:]
        loss = criterion(output, labelsV)
        test_loss += loss.data
        failures += torch.sum(torch.abs(output.data[:, -1] - labels[:, -1]) > args.threshold)
    test_loss /= num_batches
    test_failrate = float(failures)/float(num_batches * args.test_batch_size)
    return test_loss, test_failrate

def oa_test(net, criterion1, criterion2, data):
    it = iter(data)
    num_batches = len(data)
    test_loss = 0
    failures = 0
    total_object_detection_correct = 0
    total_left_avoidance_correct = 0
    total_right_avoidance_correct = 0
    net.eval()
    for bnum in tqdm(range(num_batches)):
        batch_data = next(it)
        imgs = batch_data['image'].cuda()
        status_inputs = batch_data['status_input'].type('torch.FloatTensor').cuda()
        labels = batch_data['label'].type('torch.FloatTensor').cuda()
        imgsV = Variable(imgs)
        status_inputsV = Variable(status_inputs)
        labels1 = labels[:, :net.aux_labels_count]
        labels2 = labels[:, net.aux_labels_count:]
        labels1V = Variable(labels1)
        labels2V = Variable(labels2)
        inputsV = [imgsV, status_inputsV]
        output = net(inputsV)
        if args.train_patch:
            output = output[:, 2*net.labels_count:]
        output1 = output[:, :net.aux_labels_count]
        output1_sigmoid = torch.sigmoid(output1)
        output2 = output[:, net.aux_labels_count:]
        loss1 = criterion1(output1, labels1V)
        loss2 = criterion2(output2, labels2V)
        loss = loss1 + loss2

        test_loss += loss.data
        failures += torch.sum(torch.abs(output.data[:, -1] - labels[:, -1]) > args.threshold)

        output_decision = (output1_sigmoid.data[:, 0]>0.5).float()
        out_decision = (output_decision == labels[:, 0]).float()
        correct = out_decision.float().sum()
        total_object_detection_correct += correct

        output_decision = (output1_sigmoid.data[:, 1]>0.5).float()
        out_decision = (output_decision == labels[:, 1]).float()
        correct = out_decision.float().sum()
        total_left_avoidance_correct += correct

        output_decision = (output1_sigmoid.data[:, 2]>0.5).float()
        out_decision = (output_decision == labels[:, 2]).float()
        correct = out_decision.float().sum()
        total_right_avoidance_correct += correct

    test_loss /= num_batches
    test_failrate = float(failures)/float(num_batches * args.batch_size)
    total_object_detection_correct = float(total_object_detection_correct)/float(num_batches * args.batch_size)
    total_left_avoidance_correct = float(total_left_avoidance_correct)/float(num_batches * args.batch_size)
    total_right_avoidance_correct = float(total_right_avoidance_correct)/float(num_batches * args.batch_size)
    object_detection_failrate = 1.0 - total_object_detection_correct
    left_avoidance_failrate = 1.0 - total_left_avoidance_correct
    right_avoidance_failrate = 1.0 - total_right_avoidance_correct
    return test_loss, test_failrate, object_detection_failrate, left_avoidance_failrate, right_avoidance_failrate


if __name__ == '__main__':
    main()
