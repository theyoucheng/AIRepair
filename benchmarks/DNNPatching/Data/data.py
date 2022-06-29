import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import pandas as pd
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from Pipeline.options import args


class DATA():
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.transform = transforms.ToTensor()
        self.csv_file = pd.read_csv(self.csv_path, header=0)
        self.imgCh = 3
        self.imgH = 128
        self.imgW = 128

    def __len__(self):
        return len(self.csv_file) - 1

    def __getitem__(self, idx):
        img_loc = self.csv_file.iloc[idx]['image_path']
        img = Image.open(img_loc)
        if self.transform:
            img = img.resize((self.imgW, self.imgH), PIL.Image.ANTIALIAS)
            img = self.transform(img)

        left_lane_change = self.csv_file.iloc[idx]['left_lane_change_activated']
        right_lane_change = self.csv_file.iloc[idx]['right_lane_change_activated']
        lane_change_part = self.csv_file.iloc[idx]['lane_change_second_half']
        if lane_change_part == -1 and left_lane_change == 1:
            llx1 = 1
            llx2 = 0
            rrx1 = 0
            rrx2 = 0
        elif lane_change_part == 1 and left_lane_change == 1:
            llx1 = 0
            llx2 = 1
            rrx1 = 0
            rrx2 = 0
        elif lane_change_part == -1 and right_lane_change == 1:
            llx1 = 0
            llx2 = 0
            rrx1 = 1
            rrx2 = 0
        elif lane_change_part == 1 and right_lane_change == 1:
            llx1 = 0
            llx2 = 0
            rrx1 = 0
            rrx2 = 1
        else:
            llx1 = 0
            llx2 = 0
            rrx1 = 0
            rrx2 = 0

        status_input = [llx1, llx2, rrx1, rrx2]
        status_input = torch.from_numpy(np.asarray(status_input))
        lc_status = float(self.csv_file.iloc[idx]['lc_status'])
        steering_command = float(self.csv_file.iloc[idx]['steering_command'])

        if args.task == 'oa':
            if self.csv_file.iloc[idx]['lane_change_second_half'] == -1 and 'oa' in img_loc:
                object_close = 1
            else:
                object_close = 0
            left_avoidance = self.csv_file.iloc[idx]['left_lane_change_activated'] * object_close
            right_avoidance = self.csv_file.iloc[idx]['right_lane_change_activated'] * object_close

            label = [object_close, left_avoidance, right_avoidance, lc_status, steering_command]
        else:
            label = [lc_status, steering_command]

        label = torch.from_numpy(np.asarray(label))

        sample = {'image':img, 'status_input':status_input, 'label':label}
        return sample


def TrainDataLoader(valid_size=0.2,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=False,
                    drop_last=True):

    if args.task == 'lc':
        if args.train_patch:
            csv_path = '/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/lf-data-v2-lc-town03-polyline-corrected2-cut-endcorrected.csv'
        elif args.train_base:
            csv_path = '/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/lf-data-v2-lc-town03-polyline-corrected2-cut-endcorrected.csv'
    elif args.task == 'oa':
        if args.train_patch:
            csv_path = '/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/lf-data-v2-oa-town03-polyline-nostops-corrected-v2-oa-town03-polyline-nostops-newversion-corrected-lf-data-v2-newversion-corrected.csv'
        elif args.train_base:
            csv_path = '/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/lf-data-v2-oa-town03-polyline-nostops-corrected-v2-oa-town03-polyline-nostops-newversion-corrected-lf-data-v2-newversion-corrected.csv'
    elif args.task == 'lc+oa' 
        if args.train_patch:
            csv_path = '/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/lf-data-v2-oa-town03-polyline-nostops-corrected-v2-oa-town03-polyline-nostops-newversion-corrected-lf-data-v2-newversion-corrected-lc_town03_polyline_corrected2_cut_endcorrected.csv'
        elif args.train_base:
            csv_path = '/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/lf-data-v2-oa-town03-polyline-nostops-corrected-v2-oa-town03-polyline-nostops-newversion-corrected-lf-data-v2-newversion-corrected-lc_town03_polyline_corrected2_cut_endcorrected.csv'
    else:
        csv_path = '/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/lf-data-v2.csv'
            

    dataset = DATA(csv_path)
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * len(indices)))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataloader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=drop_last)

    valid_dataloader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=valid_sampler,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=drop_last)

    return train_dataloader, valid_dataloader

def TestDataLoader(csv_path):
    test_dataset = DATA(csv_path)
    num_test = len(test_dataset)
    indices = list(range(int(num_test)))
    test_sampler = SequentialSampler(indices)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 sampler=test_sampler,
                                 num_workers=1,
                                 pin_memory=False,
                                 drop_last=True)
    return test_dataloader

#Datasets
trainData, trainSubset = TrainDataLoader()
lf_town03_validData =  TestDataLoader('/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/lf-data-v2-test.csv')
lc_town03_validData =  TestDataLoader('/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/lc_town03_polyline_test_corrected2_endcorrected_cut.csv')
oa_town03_validData = TestDataLoader('/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/oa-town03-polyline-nostops-test-corrected.csv')
if args.task == 'lc+oa'
    oa_town03_validData = TestDataLoader('/home/apoorva/Patching_repo/Patching/DataLoaders/csvfiles/lc_town03_polyline_test_corrected2_endcorrected_cut-oa-town03-polyline-nostops-test-corrected.csv')
