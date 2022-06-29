import torch
import torch.nn.functional as F
import torchvision.utils as utils
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Architecture.double_patch_model import net, optimizer, scheduler
from Pipeline.options import args
from DataLoaders.data import trainData, trainSubset, lf_town03_validData, lc_town03_validData, oa_town03_validData


# LF and LC train function

criterion = torch.nn.MSELoss()

def train(epoch, save_dir):
    it = iter(trainData)
    num_batches = len(trainData)
    epoch_loss = 0
    net.train()
    failures = 0
    str_values = torch.FloatTensor(args.batch_size * num_batches, 2).fill_(0)
    for bnum in tqdm(range(num_batches)):
        net.zero_grad()
        batch_data = next(it)
        imgs = batch_data['image'].cuda()
        status_inputs = batch_data['status_input'].type('torch.FloatTensor').cuda()
        labels = batch_data['label'].type('torch.FloatTensor').cuda()
        imgsV = Variable(imgs)
        status_inputsV = Variable(status_inputs)
        labelsV = Variable(labels)
        inputsV = [imgsV, status_inputsV]
        output = net(inputsV)
        if args.test_main_output:
            main_output_before = output[:, :net.labels_count]
        if args.train_patch:
            output = output[:, net.labels_count:]
        loss = criterion(output, labelsV)
        loss.backward()
        optimizer.step()

        if args.test_main_output:
            output = net(inputsV)
            main_output_after = output[:, :net.labels_count]
            diff = torch.sum(torch.abs(main_output_before - main_output_after))
            print(diff)
            if diff != 0:
                sys.stop()

        epoch_loss += loss.data
        failures += torch.sum(torch.abs(output.data[:, -1] - labels[:, -1]) > args.threshold)
        str_values.narrow(0, bnum * args.batch_size, args.batch_size).select(1, 0).copy_(labels[:, -1])
        str_values.narrow(0, bnum * args.batch_size, args.batch_size).select(1, 1).copy_(output.data[:, -1])
    # TODO make sure to return loss per sample not batch 
    epoch_loss /= num_batches
    scheduler.step()
    failures = float(failures)/float(num_batches * args.batch_size)
    np.save(save_dir+'tr_str_values_'+str(epoch)+'.npy', str_values.numpy())
    torch.save(net.state_dict(), save_dir+'net_'+str(epoch)+'.pth')
    print('epoch '+str(epoch)+' loss: ', epoch_loss)
    return epoch_loss, failures



# LF and LC validation function

def validate(epoch, save_dir, task):
    if task == 'lf':
        validData = lf_town03_validData
    elif task == 'lc':
        validData = lc_town03_validData
    #elif task == 'oa':
    #    validData = oa_town03_validData

    it = iter(validData)
    num_batches = len(validData)
    epoch_loss = 0
    failures = 0
    str_values = torch.FloatTensor(args.batch_size * num_batches, 2).fill_(0)
    net.eval()
    for bnum in tqdm(range(num_batches)):
        #if bnum == 2:
        #    break
        batch_data = next(it)
        imgs = batch_data['image'].cuda()
        status_inputs = batch_data['status_input'].type('torch.FloatTensor').cuda()
        labels = batch_data['label'].type('torch.FloatTensor').cuda()
        imgsV = Variable(imgs)
        status_inputsV = Variable(status_inputs)
        labelsV = Variable(labels)
        #inputsV = [imgsV]
        inputsV = [imgsV, status_inputsV]
        output = net(inputsV)
        if args.train_patch:
            output = output[:, net.labels_count:]

        loss = criterion(output, labelsV)
        epoch_loss += loss.data
        failures += torch.sum(torch.abs(output.data[:, -1] - labels[:, -1]) > args.threshold)
        str_values.narrow(0, bnum * args.batch_size, args.batch_size).select(1, 0).copy_(labels[:, -1])
        str_values.narrow(0, bnum * args.batch_size, args.batch_size).select(1, 1).copy_(output.data[:, -1])
    epoch_loss /= num_batches
    failures = float(failures)/float(num_batches * args.batch_size)
    np.save(save_dir+'va_str_values_'+str(epoch)+'_task_'+task+'.npy', str_values.numpy())
    print('validation loss for epoch '+str(epoch)+' task '+task+' : ', epoch_loss)
    return epoch_loss, failures


# LF and LC save_signals function

def save_signals(epoch, task):
    if task == 'lf':
        validData = lf_town03_validData
    elif task == 'lc':
        validData = lc_town03_validData
        
    it = iter(validData)
    num_batches = int(len(validData))
    net.eval()
    labels_list = []
    predictions_list = []
    lc_status_list = []
    lc_status_pred_list = []
    llx1_indicator_list = []
    llx2_indicator_list = []
    rrx1_indicator_list = []
    rrx2_indicator_list = []
    for bnum in tqdm(range(num_batches)):
        #if bnum == 2:
        #    break
        batch_data = next(it)
        imgs = batch_data['image'].cuda()
        status_inputs = batch_data['status_input'].type('torch.FloatTensor').cuda()
        labels = batch_data['label'].type('torch.FloatTensor')[:, 1].numpy().tolist()
        labels_list.extend(labels)
        lc_status = batch_data['label'].type('torch.FloatTensor')[:, 0].numpy().tolist()
        lc_status_list.extend(lc_status)

        llx1_indicator = status_inputs[:, 0].cpu().numpy().tolist()
        llx2_indicator = status_inputs[:, 1].cpu().numpy().tolist()
        rrx1_indicator = status_inputs[:, 2].cpu().numpy().tolist()
        rrx2_indicator = status_inputs[:, 3].cpu().numpy().tolist()
        llx1_indicator_list.extend(llx1_indicator)
        llx2_indicator_list.extend(llx2_indicator)
        rrx1_indicator_list.extend(rrx1_indicator)
        rrx2_indicator_list.extend(rrx2_indicator)

        imgsV = Variable(imgs)
        status_inputsV = Variable(status_inputs)
        inputsV = [imgsV, status_inputsV]
        output = net(inputsV)

        if args.train_patch:
            output = output[:, net.labels_count:]

        str_output = output.data[:, -1].cpu().numpy().tolist()
        lc_status_pred = output.data[:, 0].cpu().numpy().tolist()
        predictions_list.extend(str_output)
        lc_status_pred_list.extend(lc_status_pred)

    fig, axs = plt.subplots(8, figsize=(7, 14))
    fig.suptitle('Consecutive frames on x-axis')
    axs[0].plot(labels_list)
    axs[0].set_title('steering labels')
    #axs[0].set_ylim([-1, 1])
    axs[1].plot(predictions_list)
    #axs[1].set_ylim([-1, 1])
    axs[1].set_title('predicted steering commands')
    axs[2].plot(lc_status_list)
    axs[2].set_title('lane change status label')
    axs[2].set_ylim([-0.1, 1])
    axs[3].plot(lc_status_pred_list)
    axs[3].set_title('predicted lane change status')
    axs[3].set_ylim([-0.1, 1])
    axs[4].plot(llx1_indicator_list)
    axs[4].set_title('first half of left lc indicator input')
    axs[5].plot(llx2_indicator_list)
    axs[5].set_title('second half ofleft lc indicator input')
    axs[6].plot(rrx1_indicator_list)
    axs[6].set_title('first half of right lc indicator input')
    axs[7].plot(rrx2_indicator_list)
    axs[7].set_title('second half of right lc indicator input')
    #fig.xlabel('frame number')
    #fig.tight_layout(pad=7.0)
    for ax in fig.get_axes():
        ax.label_outer()
    
    fig.savefig(args.output_dir+'va_signals_epoch'+str(epoch)+'_task_'+task+'.png')
    plt.close('all')


# OA train function

criterion1 = torch.nn.BCEWithLogitsLoss()
criterion2 = torch.nn.MSELoss()

def train_oa(epoch, save_dir):
    it = iter(trainData)
    num_batches = len(trainData)
    epoch_loss = 0
    net.train()
    failures = 0
    total_object_detection_correct = 0
    total_left_avoidance_correct = 0
    total_right_avoidance_correct = 0

    str_values = torch.FloatTensor(args.batch_size * num_batches, 2).fill_(0)
    object_detection_values = torch.FloatTensor(args.batch_size * num_batches, 2).fill_(0)
    left_avoidance_values = torch.FloatTensor(args.batch_size * num_batches, 2).fill_(0)
    right_avoidance_values = torch.FloatTensor(args.batch_size * num_batches, 2).fill_(0)
    
    for bnum in tqdm(range(num_batches)):
        #if bnum == 2:
        #    break
        net.zero_grad()
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
        #print("output1")
        #print(output1_sigmoid, labels1)
        #print("output2")
        output2 = output[:, net.aux_labels_count:]
        #print(output2, labels2)
        #main_output_before = output1
        #sys.stop()
        loss1 = criterion1(output1, labels1V)
        loss2 = criterion2(output2, labels2V)
        loss = loss1 + loss2
        #print(loss1, loss2, loss)
        #sys.stop()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data
        failures += torch.sum(torch.abs(output.data[:, -1] - labels[:, -1]) > args.threshold)
        str_values.narrow(0, bnum * args.batch_size,
                          args.batch_size).select(1, 0).copy_(labels[:, -1])
        str_values.narrow(0, bnum * args.batch_size,
                          args.batch_size).select(1, 1).copy_(output.data[:, -1])
        object_detection_values.narrow(0, bnum * args.batch_size,
                                       args.batch_size).select(1, 0).copy_(labels[:, 0])
        object_detection_values.narrow(0, bnum * args.batch_size,
                                       args.batch_size).select(1, 1).copy_(output1_sigmoid.data[:, 0])

        output_decision = (output1_sigmoid.data[:, 0]>0.5).float()
        out_decision = (output_decision == labels[:, 0]).float()
        correct = out_decision.float().sum()
        total_object_detection_correct += correct 

        left_avoidance_values.narrow(0, bnum * args.batch_size,
                                     args.batch_size).select(1, 0).copy_(labels[:, 1])
        left_avoidance_values.narrow(0, bnum * args.batch_size,
                                     args.batch_size).select(1, 1).copy_(output1_sigmoid.data[:, 1])
        output_decision = (output1_sigmoid.data[:, 1]>0.5).float()
        out_decision = (output_decision == labels[:, 1]).float()
        correct = out_decision.float().sum()
        total_left_avoidance_correct += correct

        right_avoidance_values.narrow(0, bnum * args.batch_size,
                                     args.batch_size).select(1, 0).copy_(labels[:, 2])
        right_avoidance_values.narrow(0, bnum * args.batch_size,
                                     args.batch_size).select(1, 1).copy_(output1_sigmoid.data[:, 2])

        output_decision = (output1_sigmoid.data[:, 2]>0.5).float()
        out_decision = (output_decision == labels[:, 2]).float() 
        correct = out_decision.float().sum()
        total_right_avoidance_correct += correct

    epoch_loss /= num_batches
    scheduler.step()
    failures = float(failures)/float(num_batches * args.batch_size)
    total_object_detection_correct = float(total_object_detection_correct)/float(num_batches * args.batch_size)
    total_left_avoidance_correct = float(total_left_avoidance_correct)/float(num_batches * args.batch_size)
    total_right_avoidance_correct = float(total_right_avoidance_correct)/float(num_batches * args.batch_size)
    object_detection_failures = 1.0 - total_object_detection_correct
    left_avoidance_failures = 1.0 - total_left_avoidance_correct
    right_avoidance_failures = 1.0 - total_right_avoidance_correct
    np.save(save_dir+'tr_str_values_'+str(epoch)+'.npy', str_values.numpy())
    np.save(save_dir+'tr_obj_detetion_values_'+str(epoch)+'.npy', object_detection_values.numpy())
    np.save(save_dir+'tr_left_avoid_values_'+str(epoch)+'.npy', left_avoidance_values.numpy())
    np.save(save_dir+'tr_right_avoid_values_'+str(epoch)+'.npy', right_avoidance_values.numpy())
    torch.save(net.state_dict(), save_dir+'net_'+str(epoch)+'.pth')
    print('epoch '+str(epoch)+' loss: ', epoch_loss)
    return epoch_loss, failures, object_detection_failures, left_avoidance_failures, right_avoidance_failures


def validate_oa(epoch, save_dir, task):
    if task == 'lf':
        validData = lf_town03_validData
    elif task == 'oa':
        validData = oa_town03_validData
    elif task == 'training-subset':
        validData = trainSubset

    it = iter(validData)
    num_batches = len(validData)
    epoch_loss = 0
    failures = 0
    total_object_detection_correct = 0
    total_left_avoidance_correct = 0
    total_right_avoidance_correct = 0

    str_values = torch.FloatTensor(args.batch_size * num_batches, 2).fill_(0)
    object_detection_values = torch.FloatTensor(args.batch_size * num_batches, 2).fill_(0)
    left_avoidance_values = torch.FloatTensor(args.batch_size * num_batches, 2).fill_(0)
    right_avoidance_values = torch.FloatTensor(args.batch_size * num_batches, 2).fill_(0)

    net.eval()
    for bnum in tqdm(range(num_batches)):
        #if bnum == 2:
        #    break
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
        epoch_loss += loss.data

        failures += torch.sum(torch.abs(output.data[:, -1] - labels[:, -1]) > args.threshold)
        str_values.narrow(0, bnum * args.batch_size,
                          args.batch_size).select(1, 0).copy_(labels[:, -1])
        str_values.narrow(0, bnum * args.batch_size,
                          args.batch_size).select(1, 1).copy_(output.data[:, -1])
        object_detection_values.narrow(0, bnum * args.batch_size,
                                       args.batch_size).select(1, 0).copy_(labels[:, 0])
        object_detection_values.narrow(0, bnum * args.batch_size,
                                       args.batch_size).select(1, 1).copy_(output1_sigmoid.data[:, 0])

        output_decision = (output1_sigmoid.data[:, 0]>0.5).float()
        out_decision = (output_decision == labels[:, 0]).float()
        correct = out_decision.float().sum()
        total_object_detection_correct += correct

        left_avoidance_values.narrow(0, bnum * args.batch_size,
                                     args.batch_size).select(1, 0).copy_(labels[:, 1])
        left_avoidance_values.narrow(0, bnum * args.batch_size,
                                     args.batch_size).select(1, 1).copy_(output1_sigmoid.data[:, 1])
        output_decision = (output1_sigmoid.data[:, 1]>0.5).float()
        out_decision = (output_decision == labels[:, 1]).float()
        correct = out_decision.float().sum()
        total_left_avoidance_correct += correct

        right_avoidance_values.narrow(0, bnum * args.batch_size,
                                     args.batch_size).select(1, 0).copy_(labels[:, 2])
        right_avoidance_values.narrow(0, bnum * args.batch_size,
                                     args.batch_size).select(1, 1).copy_(output1_sigmoid.data[:, 2])

        output_decision = (output1_sigmoid.data[:, 2]>0.5).float()
        out_decision = (output_decision == labels[:, 2]).float()
        correct = out_decision.float().sum()
        total_right_avoidance_correct += correct

    epoch_loss /= num_batches
    failures = float(failures)/float(num_batches * args.batch_size)
    total_object_detection_correct = float(total_object_detection_correct)/float(num_batches * args.batch_size)
    total_left_avoidance_correct = float(total_left_avoidance_correct)/float(num_batches * args.batch_size)
    total_right_avoidance_correct = float(total_right_avoidance_correct)/float(num_batches * args.batch_size)
    object_detection_failures = 1.0 - total_object_detection_correct
    left_avoidance_failures = 1.0 - total_left_avoidance_correct
    right_avoidance_failures = 1.0 - total_right_avoidance_correct
    np.save(save_dir+'va_str_values_'+str(epoch)+'_task_'+task+'.npy', str_values.numpy())
    np.save(save_dir+'va_obj_detetion_values_'+str(epoch)+'_task_'+task+'.npy', object_detection_values.numpy())
    np.save(save_dir+'va_left_avoid_values_'+str(epoch)+'_task_'+task+'.npy', left_avoidance_values.numpy())
    np.save(save_dir+'va_right_avoid_values_'+str(epoch)+'_task_'+task+'.npy', right_avoidance_values.numpy())
    print('validation loss for epoch '+str(epoch)+' task '+task+' : ', epoch_loss)
    return epoch_loss, failures, object_detection_failures, left_avoidance_failures, right_avoidance_failures


def save_signals_oa(epoch, task):

    #load_state = torch.load(args.output_dir+'net_'+str(epoch)+'.pth')
    #net.load_state_dict(load_state)

    if task == 'lf':
        validData = lf_town03_validData
    elif task == 'oa':
        validData = oa_town03_validData
    elif task == 'training-subset':
        validData = trainSubset

    it = iter(validData)
    num_batches = int(len(validData))
    net.eval()
    object_detected_labels_list = []
    left_avoidance_labels_list = []
    right_avoidance_labels_list = []
    llx1_labels_list = []
    llx2_labels_list = []
    rrx1_labels_list = []
    rrx2_labels_list = []
    llx1_pred_list = []
    llx2_pred_list = []
    rrx1_pred_list = []
    rrx2_pred_list = []
    object_detected_pred_list = []
    left_avoidance_pred_list = []
    right_avoidance_pred_list = []
    steering_labels_list = []
    steering_pred_list = []
    lc_status_labels_list = []
    lc_status_pred_list = []

    for bnum in tqdm(range(num_batches)):
        #if bnum == 2:
        #    break
        batch_data = next(it)
        imgs = batch_data['image'].cuda()
        status_inputs = batch_data['status_input'].type('torch.FloatTensor').cuda()

        ####
        llx1_indicator = status_inputs[:, 0].cpu().numpy().tolist()
        llx2_indicator = status_inputs[:, 1].cpu().numpy().tolist()
        rrx1_indicator = status_inputs[:, 2].cpu().numpy().tolist()
        rrx2_indicator = status_inputs[:, 3].cpu().numpy().tolist()
        llx1_labels_list.extend(llx1_indicator)
        llx2_labels_list.extend(llx2_indicator)
        rrx1_labels_list.extend(rrx1_indicator)
        rrx2_labels_list.extend(rrx2_indicator)
        ####

        object_close = batch_data['label'].type('torch.FloatTensor')[:, 0].numpy().tolist()
        object_detected_labels_list.extend(object_close)
        la_labels = batch_data['label'].type('torch.FloatTensor')[:, 1].numpy().tolist()
        left_avoidance_labels_list.extend(la_labels)
        ra_labels = batch_data['label'].type('torch.FloatTensor')[:, 2].numpy().tolist()
        right_avoidance_labels_list.extend(ra_labels)
        steering_labels = batch_data['label'].type('torch.FloatTensor')[:, -1].numpy().tolist()
        steering_labels_list.extend(steering_labels)
        lc_status_labels = batch_data['label'].type('torch.FloatTensor')[:, 3].numpy().tolist()
        lc_status_labels_list.extend(lc_status_labels)


        imgsV = Variable(imgs)
        status_inputsV = Variable(status_inputs)
        inputsV = [imgsV, status_inputsV]
        output = net(inputsV)
        if args.train_patch:
            output = output[:, 2*net.labels_count:]
        output1 = output[:, :net.aux_labels_count]
        output1_sigmoid = torch.sigmoid(output1)

        object_close_preds = output1_sigmoid.data[:, 0].cpu().numpy().tolist()
        left_avoidance_preds = output1_sigmoid.data[:, 1].cpu().numpy().tolist()
        right_avoidance_preds = output1_sigmoid.data[:, 2].cpu().numpy().tolist()
        lc_status_preds = output.data[:, 3].cpu().numpy().tolist()
        steering_output_preds = output.data[:, 4].cpu().numpy().tolist()

        for idx in range(args.batch_size):
            if left_avoidance_preds[idx] > 0.5:
                if object_close_preds[idx] > 0.5:
                    llx1_pred_list.append(1)
                    llx2_pred_list.append(0)
                    rrx1_pred_list.append(0)
                    rrx2_pred_list.append(0)
                else:
                    llx1_pred_list.append(0)
                    llx2_pred_list.append(1)
                    rrx1_pred_list.append(0)
                    rrx2_pred_list.append(0)
            elif right_avoidance_preds[idx] > 0.5:
                if object_close_preds[idx] > 0.5:
                    llx1_pred_list.append(0)
                    llx2_pred_list.append(0)
                    rrx1_pred_list.append(1)
                    rrx2_pred_list.append(0)
                else:
                    llx1_pred_list.append(0)
                    llx2_pred_list.append(0)
                    rrx1_pred_list.append(0)
                    rrx2_pred_list.append(1)
            else:
                llx1_pred_list.append(0)
                llx2_pred_list.append(0)
                rrx1_pred_list.append(0)
                rrx2_pred_list.append(0)


        object_detected_pred_list.extend(object_close_preds)
        left_avoidance_pred_list.extend(left_avoidance_preds)
        right_avoidance_pred_list.extend(right_avoidance_preds)
        steering_pred_list.extend(steering_output_preds)
        lc_status_pred_list.extend(lc_status_preds)

    fig, axs = plt.subplots(18, figsize=(10, 24))
    fig.suptitle('Consecutive frames on x-axis')
    axs[0].plot(object_detected_labels_list)
    axs[0].set_ylim([-0.5, 1.5])
    axs[0].set_title('object detected label')
    axs[1].plot(object_detected_pred_list)
    axs[1].set_ylim([-0.5, 1.5])
    axs[1].set_title('object detected predictions')
    axs[2].plot(left_avoidance_labels_list)
    axs[2].set_ylim([-0.5, 1.5])
    axs[2].set_title('left avoidance labels')
    axs[3].plot(left_avoidance_pred_list)
    axs[3].set_ylim([-0.5, 1.5])
    axs[3].set_title('left avoidance predictions')
    axs[4].plot(right_avoidance_labels_list)
    axs[4].set_ylim([-0.5, 1.5])
    axs[4].set_title('right avoidance labels')
    axs[5].plot(right_avoidance_pred_list)
    axs[5].set_ylim([-0.5, 1.5])
    axs[5].set_title('right avoidance predictions')
    axs[6].plot(llx1_labels_list)
    axs[6].set_title('first half of left lc indicator label')
    axs[7].plot(llx1_pred_list)
    axs[7].set_title('first half of left lc indicator predictions')
    axs[8].plot(llx2_labels_list)
    axs[8].set_title('second half ofleft lc indicator label')
    axs[9].plot(llx2_pred_list)
    axs[9].set_title('second half of left lc indicator predictions')
    axs[10].plot(rrx1_labels_list)
    axs[10].set_title('first half of right lc indicator label')
    axs[11].plot(rrx1_pred_list)
    axs[11].set_title('first half of right lc indicator predictions')
    axs[12].plot(rrx2_labels_list)
    axs[12].set_title('second half of right lc indicator label')
    axs[13].plot(rrx2_pred_list)
    axs[13].set_title('second half of right lc indicator predictions')
    axs[14].plot(steering_labels_list)
    axs[14].set_title('steering labels')
    #axs[14].set_ylim([-1, 1])
    axs[15].plot(steering_pred_list)
    #axs[15].set_ylim([-1, 1])
    axs[15].set_title('predicted steering commands')
    axs[16].plot(lc_status_labels_list)
    axs[16].set_ylim([-0.1, 1.0])
    axs[16].set_title('lane change status label')
    axs[17].plot(lc_status_pred_list)
    axs[17].set_ylim([-0.1, 1.0])
    axs[17].set_title('predicted lane change status')
    for ax in fig.get_axes():
        ax.label_outer()
    fig.savefig(args.output_dir+'va_signals_oa_epoch'+str(epoch)+'_task_'+task+'.png')
    plt.close('all')
