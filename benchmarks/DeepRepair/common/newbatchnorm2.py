import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import uniform
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter


class dnnrepair_BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, weight, bias, running_mean, running_var, target_ratio=1, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(dnnrepair_BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.target_ratio = 1-target_ratio
        self.weight = weight
        self.bias = bias
        self.running_mean = running_mean
        self.running_var = running_var

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        e = exponential_average_factor
        # calculate running estimates
        if self.training:
            half_len = input.size(0)//2
            original_half = input[:half_len]
            target_half = input[half_len:]  # target input
            target_mean = target_half.mean([0, 2, 3])
            # use biased var in train
            target_var = target_half.var([0, 2, 3], unbiased=False)
            n1 = target_half.numel() / target_half.size(1)

            mean = original_half.mean([0, 2, 3])
            # use biased var in train
            var = original_half.var([0, 2, 3], unbiased=False)
            n2 = original_half.numel() / original_half.size(1)
            '''
            with torch.no_grad():
                if self.target_ratio != 0:
                    self.running_mean.copy_(e * mean + (1-e) * e * target_mean + (1-e) ** 2 * self.running_mean)
                    # update running_var with unbiased var
                    self.running_var.copy_(e * n2 / (n2 - 1) * var + (1 - e) * e * n1 / (n1 - 1) * target_var  +  (1 - e) ** 2 * self.running_var)
                else:
                    self.running_mean.copy_(e * mean + (1 - e) * self.running_mean)
                    # update running_var with unbiased var
                    self.running_var.copy_(e * var * n2 /(n2 - 1) + (1 - e) * self.running_var)
            '''
            with torch.no_grad():
                if self.target_ratio != 0:
                    #e = 0.19
                    self.running_mean.copy_(e * ( (1-self.target_ratio) * mean + self.target_ratio * target_mean) + (1-e) * self.running_mean)
                    # update running_var with unbiased var
                    self.running_var.copy_(e * ( (1-self.target_ratio) * n2 / (n2 - 1) * var + self.target_ratio * n1 / (n1 - 1) * target_var) + (1-e) * self.running_var)
                else:
                    #e = 0.1
                    self.running_mean.copy_(e * mean + (1 - e) * self.running_mean)
                    # update running_var with unbiased var
                    self.running_var.copy_(e * var * n2 /(n2 - 1) + (1 - e) * self.running_var)
        else:
            mean = self.running_mean
            var = self.running_var



        input = (input - mean[None, :, None, None]) / \
            (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None,
                                        None] + self.bias[None, :, None, None]

        return input
