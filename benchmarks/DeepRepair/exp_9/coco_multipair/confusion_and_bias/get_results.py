import numpy as np
import argparse


def compute_confusion(confusion_matrix, first, second):
    confusion = 0
    if (first, second) in confusion_matrix:
        confusion += confusion_matrix[(first, second)]
    
    if (second, first) in confusion_matrix:
        confusion += confusion_matrix[(second, first)]
    return confusion/2

def compute_bias(confusion_matrix, first, second, third):
    return abs(compute_confusion(confusion_matrix, first, second) - compute_confusion(confusion_matrix, first, third))

def draw_bias_graph(pretrain, repair, first, second, third):
    print("draw bias graph")

    information = np.load(pretrain, allow_pickle=True, encoding = 'latin1')
    information2 = np.load(repair, allow_pickle=True, encoding = 'latin1')
    nature_accuracy = []
    bias = []

    print("pretrain information")
    for i in information:
        acc = i["accuracy"]
#        print(acc)
        nature_accuracy.append(acc)
        b = compute_bias(i["confusion"], first, second, third)
        print(b)
        bias.append(b)

    print("repair information")
    for i in information2:
        acc = i["accuracy"]
#        print(acc)
        nature_accuracy.append(acc)
        b = compute_bias(i["confusion"], first, second, third)
        print(b)
        bias.append(b)

    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    plt.vlines(len(information) - 1 , 0, max(nature_accuracy), linestyles ="dotted", colors ="k")
    x = range(len(nature_accuracy))
    ax.plot(x, nature_accuracy, 'r-', label="accuracy")
    legend2 = "bias between " + second + " and " + third + ",\n given " + first
    ax.plot(x, bias, 'g-', label=legend2)
    plt.ylabel("accuracy/bias")
    plt.xlabel("epoch")
    legend = ax.legend(loc='best', shadow=True, fontsize=14)
    plt.savefig("coco_bias.pdf", bbox_inches='tight')
    print("coco_bias.pdf saved")
    plt.show()


def draw_confusion_graph(pretrain, repair, first, second):
    print("draw confusion graph")
    
    information = np.load(pretrain, allow_pickle=True, encoding = 'latin1')
    information2 = np.load(repair, allow_pickle=True, encoding = 'latin1')
    nature_accuracy = []
    confusion = []

    print("pretrain information")
    for i in information:
        acc = i["accuracy"]
        print(acc)
        nature_accuracy.append(acc)
        confusion.append(compute_confusion(i["confusion"], first, second))

    print("repair information")
    for i in information2:
        acc = i["accuracy"]
        print(acc)
        nature_accuracy.append(acc)
        confusion.append(compute_confusion(i["confusion"], first, second))

    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    x = range(len(nature_accuracy))
    ax.plot(x, nature_accuracy, 'r-', label="accuracy")
    legend2 = "confusion between " + first + " and " + second
    ax.plot(x, confusion, 'g-', label=legend2)
    plt.ylabel("accuracy/bias")
    plt.xlabel("epoch")
    legend = ax.legend(loc='best', shadow=True, fontsize=14)
    plt.savefig("coco_confusion.pdf", bbox_inches='tight')
    print("coco_confusion.pdf saved")
    plt.show()


if __name__ == '__main__':
    print("Results")
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--original_model', default='/set/your/model/path', type=str, metavar='PATH')
    parser.add_argument(
    '--repaired_model', default='/set/your/model/path', type=str, metavar='PATH')
    parser.add_argument('--confusion', help='Check model accuracy', action='store_true')
    parser.add_argument('--bias', help='Check model accuracy', action='store_true')
    parser.add_argument('--first', default="person", type=str,
                        help='first object index')
    parser.add_argument('--second', default="bus", type=str,
                        help='second object index')
    parser.add_argument('--third', default="clock", type=str,
                        help='third object index')
    args = parser.parse_args()
    assert os.path.isfile(args.pretrained)
    print(args)
    if args.confusion:
        draw_confusion_graph(args.original_model, args.repaired_model, args.first, args.second)
    if args.bias:
        draw_bias_graph(args.original_model, args.repaired_model, args.first, args.second, args.third)
