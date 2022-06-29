import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Pipeline.options import args
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.status_inputs_count = args.status_inputs_count
        self.train_patch = args.train_patch
        self.roi_channels_count = 3
        self.labels_count = 2
        self.dropout_rate = 0.2
        self.str_dropout_rate = 0.1
        self.patch_dropout_rate = 0.1

        #self.channel_counts = [self.roi_channels_count, 12, 12, 24, 24, 48, 64]
        self.channel_counts = [self.roi_channels_count, 8, 8, 16, 16, 32, 64]
        #self.fc_features = [self.status_inputs_count + self.channel_counts[-1]*2*2,
        #                    64, self.labels_count]
        self.fc_features = [self.status_inputs_count + self.channel_counts[-1]*2*2,
                            32, self.labels_count]

        if args.task == 'oa':
            self.aux_labels_count = 3
            self.fc_aux_features = [self.channel_counts[-1]*2*2, 32, self.aux_labels_count] # 32, self.aux_labels_count]

        # TODO: add argument to select this configuration
        #exp01
        self.patch_output_channels_counts = [4, 4, 8, 8, 16, 32]
        self.patch_channels_count = [3, 4+8, 4+8, 8+16, 8+16, 16+32, 32+64]
        #exp02
        #self.patch_output_channels_counts = [0, 0, 0, 0, 0, 2]
        #self.patch_channels_count = [0, 0, 0, 0, 0, 32, 2+64]
        #exp03
        #self.patch_output_channels_counts = [0, 0, 0, 0, 0, 0]
        #self.patch_channels_count = [0, 0, 0, 0, 0, 0, 64]
        #exp04
        #self.patch_output_channels_counts = [2, 2, 2, 2, 2, 2]
        #self.patch_channels_count = [3, 2+8, 2+8, 2+16, 2+16, 2+32, 2+64]
        
        self.patch_fc_features_output_counts = [32, self.labels_count]
        self.patch_fc_features = [self.status_inputs_count + self.patch_channels_count[-1]*2*2,
                                  self.fc_features[1] + self.patch_fc_features_output_counts[0]]

        if args.task == 'oa':
            self.patch_fc_aux_features_output_counts = [32, self.aux_labels_count]
            self.patch_fc_aux_features = [self.patch_channels_count[-1]*2*2,
                                          self.patch_fc_aux_features_output_counts[0]]
            

            self.initialize_training_layers(self.channel_counts, self.fc_features,
                                            self.fc_aux_features)
            if args.train_patch:
                self.initialize_patch_layers(
                    self.patch_channels_count, self.patch_output_channels_counts,
                    self.patch_fc_features, self.patch_fc_features_output_counts,
                    self.patch_fc_aux_features, self.patch_fc_aux_features_output_counts
                )
        else:
            self.initialize_training_layers(self.channel_counts, self.fc_features)
            if args.train_patch:
                self.initialize_patch_layers(
                    self.patch_channels_count, self.patch_output_channels_counts,
                    self.patch_fc_features, self.patch_fc_features_output_counts
                )
            

    def resnet_subblocks(self, in_channels, out_channels, kernel_size):
        """Builds and returns residual module sub-blocks."""
        # Only odd kernel sizes supported
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError(
                "Kernel size {} not currently supported in resnet sub-blocks!".format(kernel_size))
        padding = (kernel_size - 1) // 2
        a = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(num_features=out_channels), nn.ReLU(), #TODO: make this dropout + relu
                nn.Conv2d(out_channels, out_channels, kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(num_features=out_channels))
        b = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(num_features=out_channels))
        return a, b

    def initialize_training_layers(self, channel_counts, fc_features, fc_aux_features=None):

        self.rd_activation = nn.Sequential(nn.ReLU(True), nn.Dropout(self.dropout_rate))

        self.str_rd_activation = nn.Sequential(nn.ReLU(True), nn.Dropout(self.str_dropout_rate))

        self.image_bn = nn.Sequential(nn.BatchNorm2d(channel_counts[0]))

        self.a1, self.b1 = self.resnet_subblocks(in_channels=channel_counts[0],
                                                 out_channels=channel_counts[1], kernel_size=3)
        self.a2, self.b2 = self.resnet_subblocks(in_channels=channel_counts[1],
                                                 out_channels=channel_counts[2], kernel_size=3)
        self.a3, self.b3 = self.resnet_subblocks(in_channels=channel_counts[2],
                                                 out_channels=channel_counts[3], kernel_size=3)
        self.a4, self.b4 = self.resnet_subblocks(in_channels=channel_counts[3],
                                                 out_channels=channel_counts[4], kernel_size=3)
        self.a5, self.b5 = self.resnet_subblocks(in_channels=channel_counts[4],
                                                 out_channels=channel_counts[5], kernel_size=3)

        self.fe = nn.Sequential(
            nn.Conv2d(channel_counts[5], channel_counts[6], 3, 1, padding=0),
            nn.BatchNorm2d(channel_counts[6]),
            self.rd_activation)

        fc_layers = []
        for i in range(len(fc_features)-1):
            lin_layer = nn.Linear(fc_features[i], fc_features[i+1])
            fc_layers += [lin_layer, self.str_rd_activation]
        self.fc = nn.Sequential(*fc_layers[:-1])

        if args.task == 'oa' and not args.train_patch:
            fc_aux_layers = []
            for i in range(len(fc_aux_features)-1):
                lin_layer = nn.Linear(fc_aux_features[i], fc_aux_features[i+1])
                fc_aux_layers += [lin_layer, self.rd_activation]
            self.fc_aux = nn.Sequential(*fc_aux_layers[:-1])

    def initialize_patch_layers(self, channel_counts_in, channel_counts_out,
                                fc_features_in, fc_features_out,
                                fc_aux_features_in=None, fc_aux_features_out=None):

        self.patch_rd_activation = nn.Sequential(nn.ReLU(True), nn.Dropout(self.patch_dropout_rate))

        if self.patch_output_channels_counts[0] != 0:
            self.a1_patch, self.b1_patch = self.resnet_subblocks(
                in_channels=channel_counts_in[0], out_channels=channel_counts_out[0], kernel_size=3)
        if self.patch_output_channels_counts[1] != 0:
            self.a2_patch, self.b2_patch = self.resnet_subblocks(
                in_channels=channel_counts_in[1], out_channels=channel_counts_out[1], kernel_size=3)
        if self.patch_output_channels_counts[2] != 0:
            self.a3_patch, self.b3_patch = self.resnet_subblocks(
                in_channels=channel_counts_in[2], out_channels=channel_counts_out[2], kernel_size=3)
        if self.patch_output_channels_counts[3] != 0:
            self.a4_patch, self.b4_patch = self.resnet_subblocks(
                in_channels=channel_counts_in[3], out_channels=channel_counts_out[3], kernel_size=3)
        if self.patch_output_channels_counts[4] != 0:
            self.a5_patch, self.b5_patch = self.resnet_subblocks(
                in_channels=channel_counts_in[4], out_channels=channel_counts_out[4], kernel_size=3)


        if self.patch_output_channels_counts[5] != 0:
            self.fe_patch = nn.Sequential(
                nn.Conv2d(channel_counts_in[5], channel_counts_out[5], 3, 1, padding=0),
                nn.BatchNorm2d(channel_counts_out[5]),
                self.patch_rd_activation)

        patch_fc_layers = []
        for i in range(len(fc_features_in)):
            lin_layer = nn.Linear(fc_features_in[i], fc_features_out[i])
            patch_fc_layers += [lin_layer, self.str_rd_activation]
        self.fc_patch = nn.Sequential(*patch_fc_layers[:-1])

        if args.task == 'oa':
            patch_fc_aux_layers = []
            for i in range(len(fc_aux_features_in)):
                lin_layer = nn.Linear(fc_aux_features_in[i], fc_aux_features_out[i])
                patch_fc_aux_layers += [lin_layer, self.patch_rd_activation]
            self.fc_aux_patch = nn.Sequential(*patch_fc_aux_layers[:-1])

    def forward(self, inputs):
        """
        if args.train_patch:
            for name, submodule in net.named_modules():
                if 'patch' not in name and not name ==  '':
                    #print(name)
                    #print(submodule)
                    submodule.eval()
        """
        #assert not self.a1.training
        #assert not self.rd_activation.training
        #assert self.patch_rd_activation.training
        #assert self.a1_patch.training

        x, y = inputs[0], inputs[1]

        x = self.image_bn(x)

        # Residual Block 1
        x_a = self.a1(x)
        x_b = self.b1(x)
        x_sum = x_a + x_b
        x_sum = self.rd_activation(x_sum) #F.relu(x_sum)
        if self.patch_output_channels_counts[0] != 0 and args.train_patch:
            x_patch_a = self.a1_patch(x)
            x_patch_b = self.b1_patch(x)
            x_patch_sum = x_patch_a + x_patch_b
            x_patch_sum = self.patch_rd_activation(x_patch_sum)

        # Residual Block 2
        x = x_sum
        x_a = self.a2(x)
        x_b = self.b2(x)
        x_sum = x_a + x_b
        x_sum = self.rd_activation(x_sum) #F.relu(x_sum)
        if self.patch_output_channels_counts[1] != 0 and args.train_patch:
            if self.patch_output_channels_counts[0] != 0:
                x_patch = x_patch_sum
                x_concat = torch.cat((x, x_patch), dim=1)
            else:
                x_concat = x
            x_patch_a = self.a2_patch(x_concat)
            x_patch_b = self.b2_patch(x_concat)
            x_patch_sum = x_patch_a + x_patch_b
            x_patch_sum = self.patch_rd_activation(x_patch_sum)

        # Residual Block 3
        x = x_sum
        x_a = self.a3(x)
        x_b = self.b3(x)
        x_sum = x_a + x_b
        x_sum = self.rd_activation(x_sum) #F.relu(x_sum)
        if self.patch_output_channels_counts[2] != 0 and args.train_patch:
            if self.patch_output_channels_counts[1] != 0:
                x_patch = x_patch_sum
                x_concat = torch.cat((x, x_patch), dim=1)
            else:
                x_concat = x
            x_patch_a = self.a3_patch(x_concat)
            x_patch_b = self.b3_patch(x_concat)
            x_patch_sum = x_patch_a + x_patch_b
            x_patch_sum = self.patch_rd_activation(x_patch_sum)

        # Residual Block 4
        x = x_sum
        x_a = self.a4(x)
        x_b = self.b4(x)
        x_sum = x_a + x_b
        x_sum = self.rd_activation(x_sum) #F.relu(x_sum)
        if self.patch_output_channels_counts[3] != 0 and args.train_patch:
            if self.patch_output_channels_counts[2] != 0:
                x_patch = x_patch_sum
                x_concat = torch.cat((x, x_patch), dim=1)
            else:
                x_concat = x
            x_patch_a = self.a4_patch(x_concat)
            x_patch_b = self.b4_patch(x_concat)
            x_patch_sum = x_patch_a + x_patch_b
            x_patch_sum = self.patch_rd_activation(x_patch_sum)

        # Residual Block 5
        x = x_sum
        x_a = self.a5(x)
        x_b = self.b5(x)
        x_sum = x_a + x_b
        x_sum = self.rd_activation(x_sum) #F.relu(x_sum)
        if self.patch_output_channels_counts[4] != 0 and args.train_patch:
            if self.patch_output_channels_counts[3] != 0:
                x_patch = x_patch_sum
                x_concat = torch.cat((x, x_patch), dim=1)
            else:
                x_concat = x
            x_patch_a = self.a5_patch(x_concat)
            x_patch_b = self.b5_patch(x_concat)
            x_patch_sum = x_patch_a + x_patch_b
            x_patch_sum = self.patch_rd_activation(x_patch_sum)

        # Feature Extraction
        x = x_sum
        x_fe = self.fe(x)
        if self.patch_output_channels_counts[5] != 0 and args.train_patch:
            if self.patch_output_channels_counts[4] != 0:
                x_patch = x_patch_sum
                x_concat = torch.cat((x, x_patch), dim=1)
            else:
                x_concat = x
            x_patch_fe = self.fe_patch(x_concat)

        # Reshape for fc-layers
        x = x_fe
        x = x.reshape(-1, self.channel_counts[-1]*2*2)
        if args.train_patch and self.patch_output_channels_counts[5] != 0:
            x_patch = x_patch_fe
            x_patch = x_patch.reshape(-1, self.patch_output_channels_counts[-1]*2*2)

        if args.task == 'oa':
            if args.train_base:
                fc_aux_layers_list = list(self.fc_aux.modules())[1:]
                if args.train_patch:
                    fc_aux_patch_layers_list = list(self.fc_aux_patch.modules())[1:]
                    #print(fc_aux_patch_layers_list)

                x_fc_aux = fc_aux_layers_list[0](x)
                x_fc_aux = fc_aux_layers_list[1](x_fc_aux)
                if args.train_patch and self.patch_fc_aux_features_output_counts[0] != 0:
                    if self.patch_output_channels_counts[5] != 0:
                        x_concat = torch.cat((x, x_patch), dim=1)
                    else:
                        x_concat = x
                    x_patch_fc_aux = fc_aux_patch_layers_list[0](x_concat)
                    x_patch_fc_aux = fc_aux_patch_layers_list[1](x_patch_fc_aux)

                #x_fc_aux = fc_aux_layers_list[4](x_fc_aux)
                if args.train_patch and self.patch_fc_aux_features_output_counts[1] != 0:
                    if self.patch_fc_aux_features_output_counts[0] != 0:
                        x_concat = torch.cat((x_fc_aux, x_patch_fc_aux), dim=1)
                    else:
                        x_concat = x_fc_aux
                    #print(fc_aux_patch_layers_list[4], x_concat.shape)
                    x_patch_fc_aux = fc_aux_patch_layers_list[4](x_concat)
                x_fc_aux = fc_aux_layers_list[4](x_fc_aux)

            elif args.train_patch:
                fc_aux_patch_layers_list = list(self.fc_aux_patch.modules())[1:]
                if self.patch_fc_aux_features_output_counts[0] != 0:
                    if self.patch_output_channels_counts[5] != 0:
                        x_concat = torch.cat((x, x_patch), dim=1)
                    else:
                        x_concat = x
                    x_patch_fc_aux = fc_aux_patch_layers_list[0](x_concat)
                    x_patch_fc_aux = fc_aux_patch_layers_list[1](x_patch_fc_aux)

                if self.patch_fc_aux_features_output_counts[1] != 0:
                    if self.patch_fc_aux_features_output_counts[0] != 0:
                        x_concat = x_patch_fc_aux #torch.cat((x_patch_fc_aux), dim=1)
                    else:
                        #TODO handle error properly
                        print("Error")
                        sys.stop()
                    x_patch_fc_aux = fc_aux_patch_layers_list[4](x_concat)
 
        y = y.reshape(-1, self.status_inputs_count)
        x = torch.cat((y, x), dim=1)
 
        # TODO: Simplify with patch fc operatons
        fc_layers_list = list(self.fc.modules())[1:]
        if args.train_patch:
            fc_patch_layers_list = list(self.fc_patch.modules())[1:]

        x_fc = fc_layers_list[0](x)
        x_fc = fc_layers_list[1](x_fc)
        if args.train_patch and self.patch_fc_features_output_counts[0] != 0:
            if self.patch_output_channels_counts[5] != 0:
                x_concat = torch.cat((x, x_patch), dim=1)
            else:
                x_concat = x
            x_patch_fc = fc_patch_layers_list[0](x_concat)
            x_patch_fc = fc_patch_layers_list[1](x_patch_fc)

        x = x_fc
        x_fc = fc_layers_list[4](x) #2
        if args.train_patch and self.patch_fc_features_output_counts[1] != 0:
            if self.patch_fc_features_output_counts[0] != 0:
                x_patch = x_patch_fc
                x_concat = torch.cat((x, x_patch), dim=1)
            else:
                x_concat = x
            x_patch_fc = fc_patch_layers_list[4](x_concat)
       
        if args.task == 'oa':
            if args.train_patch:
                out = torch.cat((x_fc, x_patch_fc_aux, x_patch_fc), dim=1)
            elif args.train_base:
                out = torch.cat((x_fc_aux, x_fc), dim=1)
        else:
            if args.train_patch:
                out = torch.cat((x_fc, x_patch_fc), dim=1)
            else:
                out = x_fc
        return out


net = Model()
if args.cuda:
    net.cuda()

#print(net)

if args.train_patch:
    for name, param in net.named_parameters():
        if 'patch' not in name:
            #print(name)
            #print(np.prod(param.size()))
            param.requires_grad = False


if args.train_patch:
    load_state = torch.load(args.resume_from)
    net.load_state_dict(load_state, strict=False)


def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters()])
print('num of parameters: ', count_parameters(net))

#TODO: add arguments to set learning rate and weight decay hyper-parameters.
optimizer = optim.AdamW(net.parameters(), lr = 0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) #25

input1 = torch.Tensor(10, 3, 128, 128).cuda()
input2 = torch.Tensor(10, args.status_inputs_count).cuda()

# TODO: Remove this if condition
if not args.finetune_prunednet:
    torch.onnx.export(net, [input1, input2],
                      args.output_dir+args.task+"_model_patch_"+str(args.train_patch)+".onnx",
                      verbose=False, input_names = ['image_input', 'status_input'],
                      output_names = ['output'])
