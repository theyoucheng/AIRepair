import argparse
import numpy as np

def get_cifar100_labels():
    import pickle
    with open("/home/yuchi/data/cifar-100-python/meta", "rb") as metafile:
        meta = pickle.load(metafile, encoding='latin1')
    return meta


def top_confusions(epoch_confusion_file, n = 5):
    information = np.load(epoch_confusion_file, allow_pickle=True)
    epoch = []
    nature_accuracy = []
    dogcat_natural_confusion = []
    top_natural_confusion = []
    lr = []
    for i in information:
        epoch.append(i["epoch"])
        lr.append(i["lr"])
        nature_accuracy.append(i["accuracy"]/100)
        dogcat_natural_confusion.append(i["confusion"][(5, 3)])
        top_natural_confusion.append(i["confusion"][sorted(i["confusion"], key=i["confusion"].get, reverse=True)[0]])
    print("first stage")
    max_acc = 0
    max_index = 0
    for i in range(len(nature_accuracy)):
        if nature_accuracy[i] >= max_acc:
            max_acc = nature_accuracy[i]
            max_index = i

    print(nature_accuracy[max_index])
    print(information[max_index]["confusion"][(5, 3)])
    print(information[max_index]["confusion"][(3, 5)])
    confusion_matrix = information[max_index]["confusion"]
    avg_pair_confusion = {}
    for i in range(100):
        for j in range(i + 1, 100):
            avg_pair_confusion[(i,j)] = (confusion_matrix[(i, j)] + confusion_matrix[(j, i)])/2
    keys = sorted(avg_pair_confusion, key=avg_pair_confusion.get, reverse=True)[:n]
    for i in range(n):
        print("")
        print(str(keys[i]) + " avg: " + str(avg_pair_confusion[keys[i]]))
        print(str(keys[i]) + ": " + str(confusion_matrix[keys[i]]))
        print(str(keys[i][::-1]) + ": " + str(confusion_matrix[keys[i][::-1]]))

    #print(top_natural_confusion[max_index])

def draw_graph():

    information = np.load("./log/cifar100_resnet_1_epoch_confusion.npy", allow_pickle=True)
    nature_accuracy = []
    dogcat_confusion = []
    autotruck_confusion = []
    airship_confusion = []
    cur_acc = 0
    for i in information:
        acc = i["accuracy"]/100
        if acc > cur_acc:
            nature_accuracy.append(acc)
            cur_acc = acc
            dogcat_confusion.append((i["confusion"][(35, 98)] + i["confusion"][(98, 35)])/2)
            autotruck_confusion.append((i["confusion"][(47, 52)] + i["confusion"][(52, 47)])/2)
            airship_confusion.append((i["confusion"][(11, 46)] + i["confusion"][(46, 11)])/2)
    print("first stage")


    print(nature_accuracy[-1])
    print(information[-1]["confusion"][(5,3)])
    print(information[-1]["confusion"][(3,5)])

    print(nature_accuracy[-2])
    print(dogcat_confusion[-2])
    print(autotruck_confusion[-2])
    print(airship_confusion[-2])
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    ax.plot(nature_accuracy, dogcat_confusion, 'r-', label="girl/woman confusion")
    ax.plot(nature_accuracy, autotruck_confusion, 'g-', label="maple/oak confusion")
    ax.plot(nature_accuracy, airship_confusion, 'b-', label="boy/man confusion")
    plt.ylabel("confusion")
    plt.xlabel("natural accuracy")
    legend = ax.legend(loc='upper center', shadow=True, fontsize=12)
    plt.savefig("cifar100.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
    parser.add_argument('--npy', default='', type=str,
                        help='epoch confusion')
    
    args = parser.parse_args()
    assert os.path.isfile(args.pretrained)

    top_confusions(args.npy, 3)


    meta = get_cifar100_labels()
    for i in range(100):
        print(meta["fine_label_names"][i])
    print(meta["fine_label_names"][35])
    print(meta["fine_label_names"][98])
    #print(meta["fine_label_names"][47])
    #print(meta["fine_label_names"][52])
    #print(meta["fine_label_names"][11])
    #print(meta["fine_label_names"][46])

    #draw_graph()