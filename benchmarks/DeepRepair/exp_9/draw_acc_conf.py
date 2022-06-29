from matplotlib import pyplot as plt
coco_acc_conf_list = [
# ('w-os 0.12', 66.03, 0.005, 0.342),
# ('w-os 0.2', 66.02, 0.004, 0.236),
# ('w-os 0.3', 66.01, 0.002, 0.142),
# ('w-os 0.4', 65.99, 0.0002, 0.031),


('w-os 0.5', 0.6599, 0, 0),
('w-os 0.6', 0.6601, 0.0842, 0.0009),
('w-os 0.7', 0.6602, 0.2289, 0.0029),
('w-os 0.8', 0.6603, 0.3425, 0.0050),
('w-os 0.9', 0.6604, 0.3993, 0.0067),
('orig', 0.6604, 0.4689, 0.0074)


# ('w-bn 0.2 retrain', 66.03, 0.002, 0.315),
# ('w-bn 0.4 retrain', 65.94, 0.0002, 0.234),
# ('w-bn 0.6 retrain', 65.79, 0, 0.123),
# ('w-bn 0.8 retrain', 65.60, 0, 0.049),

# ('orig', 66.04, 0.007, 0.469)
]


cifar10_acc_conf_list = [
# ('w-os 0.2', 87.44, 0.072, 0.092),
# ('w-os 0.3', 86.68, 0.054, 0.073),
# ('w-os 0.4', 85.25, 0.03, 0.044),
# ('w-os 0.49', 80.55, 0.006, 0.005),
# ('w-os 0.5', 72.88, 0.0, 0.0),



('w-os 0.001', 0.8056,  0.018, 0.021),
('w-os 0.01', 0.8390, 0.038,0.048),
('w-os 0.1', 0.8654, 0.06, 0.082),
('w-os 0.3', 0.873, 0.073, 0.093),
('w-os 0.5', 0.8741, 0.078, 0.098),
('orig', 0.8747, 0.085, 0.107),

# ('w-bn 0.4', 82.34, 0.023, 0.078),
# ('w-bn 0.5', 79.62, 0.016, 0.067),
# ('w-bn 0.5 retrain', 80.83, 0.021, 0.061),
# ('w-bn 1.0 retrain', 67.38, 0.0, 0.005),

# ('orig', 87.47, 0.085, 0.107)
]

# color_map = {'orig': 'black', 'w-os': 'red', 'w-bn': 'blue'}
colors = ['red', 'orange', 'purple', 'blue', 'green', 'black']
# 'coco', 'cifar10'
dataset = 'cifar10'

if dataset == 'cifar10':
    acc_conf_list = cifar10_acc_conf_list
elif dataset == 'coco':
    acc_conf_list = coco_acc_conf_list

plt.gca().invert_yaxis()

for i, (method, acc, conf1, conf2) in enumerate(acc_conf_list):
    conf = (conf1+conf2)/2

    # color = 'yellow'
    # for m, color in color_map.items():
    #     if m in method:
    #         color = color_map[m]
    #         break
    color = colors[i]
    print(color)

    plt.scatter(acc*100, conf*100, marker='o', s=200, color=color, label=method)


plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title(dataset, fontsize=25)
plt.xlabel('accuracy(%)', fontsize=22)
plt.ylabel('confusion(%)', fontsize=22)
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(loc='lower left', framealpha=0.5, prop={'size': 16})
plt.savefig('acc_conf_plot_'+dataset+'.pdf')
