# import numpy as np
# from torchtext import data
# import os 
#
# '''
# https://pytorch.org/text/_modules/torchtext/datasets/sst.html
# @inproceedings{socher2013recursive,
  # title={Recursive deep models for semantic compositionality over a sentiment treebank},
  # author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and Potts, Christopher},
  # booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
  # pages={1631--1642},
  # year={2013}
# }
# '''
#
# def filter_neutral(file_path):
    # if os.path.isfile(file_path):
        # file_path=os.path.dirname(file_path)
        #
    # sh =  os.path.join(file_path,"filter.sh")
    # print ("true find the file ,",sh)
    # if not os.path.isfile(sh):
        # shell=f'''
# cd {file_path}
#
# for filename in *.txt; do mv "$filename" "bak_$filename"; done;
# cat bak_train.txt |grep -v '^(2' >train.txt
# cat bak_dev.txt |grep -v '^(2' >dev.txt
# cat bak_test.txt |grep -v '^(2' >test.txt
#
    # '''
        # with open(sh,"w") as f :
            # f.write(shell)
            #
        # import subprocess
        # list_files = subprocess.run(["sh", sh])
        # return True 
        #
    # return False 
    #
# class SST(data.Dataset):
#
    # urls = ['http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip']
    # name = 'sst'
    # dirname = 'trees'
    #
    # @staticmethod
    # def sort_key(ex):
        # return len(ex.text)
        #
    # def __init__(self, path, text_field, label_field,subtrees=False,fine_grained=False, **kwargs):
        # """Create an IMDB dataset instance given a path and fields.
        #
        # Arguments:
            # path: Path to the dataset's highest level directory
            # text_field: The field that will be used for text data.
            # label_field: The field that will be used for label data.
            # Remaining keyword arguments: Passed to the constructor of
                # data.Dataset.
        # """
        # filter_neutral(path)
        #
        # fields = [('text', text_field), ('label', label_field)]
        #
        # def get_label_str(label):
            # pre = 'very ' if fine_grained else ''
            # return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
                    # '3': 'positive', '4': pre + 'positive', None: None}[label]
        # label_field.preprocessing = data.Pipeline(get_label_str)
        # with open(os.path.expanduser(path)) as f:
            # if subtrees:
                # examples = [ex for line in f for ex in
                            # data.Example.fromtree(line, fields, True)]
            # else:
                # examples = [data.Example.fromtree(line, fields) for line in f]
                #
        # # print (type(examples),"examples","----"*8,len(examples))
        # # print (examples[0],os.path.expanduser(path))
        # # exit()
        # super(SST, self).__init__(examples, fields, **kwargs)
        #
    # @classmethod
    # def splits(cls, text_field, label_field, root='.data',
               # train='train.txt', validation='dev.txt', test='test.txt',
               # train_subtrees=False, **kwargs):
        # """Create dataset objects for splits of the IMDB dataset.
        #
        # Arguments:
            # text_field: The field that will be used for the sentence.
            # label_field: The field that will be used for label data.
            # root: Root dataset storage directory. Default is '.data'.
            # train: The directory that contains the training examples
            # test: The directory that contains the test examples
            # Remaining keyword arguments: Passed to the splits method of
                # Dataset.
        # """
        # path = cls.download(root)
        #
        # train_data = None if train is None else cls(
            # os.path.join(path, train), text_field, label_field, subtrees=train_subtrees,
            # **kwargs)
        # val_data = None if validation is None else cls(
            # os.path.join(path, validation), text_field, label_field, **kwargs)
        # test_data = None if test is None else cls(
            # os.path.join(path, test), text_field, label_field, **kwargs)
        # # return tuple(d for d in (train_data, val_data, test_data)
                     # # if d is not None)
        # return tuple(d for d in (train_data, val_data)
                     # if d is not None)
                     #
    # def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        # """Create iterator objects for splits of the SST dataset.
        #
        # Arguments:
            # batch_size: Batch_size
            # device: Device to create batches on. Use - 1 for CPU and None for
                # the currently active GPU device.
            # root: The root directory that the dataset's zip archive will be
                # expanded into; therefore the directory in whose trees
                # subdirectory the data files will be stored.
            # vectors: one of the available pretrained vectors or a list with each
                # element one of the available pretrained vectors (see Vocab.load_vectors)
            # Remaining keyword arguments: Passed to the splits method.
        # """
        # TEXT = data.Field()
        # LABEL = data.Field(sequential=False)
        #
        # train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)
        #
        # TEXT.build_vocab(train, vectors=vectors)
        # LABEL.build_vocab(train)
        #
        # return data.BucketIterator.splits(
            # (train, val, test), batch_size=batch_size, device=device)
            #
            #
            #
# class SST_LST(data.Dataset):
#
#
    # @staticmethod
    # def sort_key(ex):
        # return len(ex.text)
        #
    # def __init__(self, path, text_field, label_field, **kwargs):
        # """Create an IMDB dataset instance given a path and fields.
        #
        # Arguments:
            # path: Path to the dataset's highest level directory
            # text_field: The field that will be used for text data.
            # label_field: The field that will be used for label data.
            # Remaining keyword arguments: Passed to the constructor of
                # data.Dataset.
        # """
        #
        # # path = npy/train npy/test
        # fields = [('text', text_field), ('label', label_field)]
        # examples = []
        # # print(path)
        # temp_data = kwargs[path]
        #
        # #
        # # x_train = temp_data['train']
        # # y_train = temp_data['label']
        # for item in temp_data:
        #
            # # examples.append(data.Example.fromlist([x_train[i], 'pos' if y_train[i] == 1 else 'neg'], fields))
            # ex = data.Example()
            # setattr(ex, 'text', item[1])
            # setattr(ex, 'label', item[0])
            # examples.append(ex)
        # super(SST_LST, self).__init__(examples, fields)
        #
    # @classmethod
    # def splits(cls, text_field, label_field, root='.data',
               # train='train', test='test', **kwargs):
        # """Create dataset objects for splits of the IMDB dataset.
        #
        # Arguments:
            # text_field: The field that will be used for the sentence.
            # label_field: The field that will be used for label data.
            # root: Root dataset storage directory. Default is '.data'.
            # train: The directory that contains the training examples
            # test: The directory that contains the test examples
            # Remaining keyword arguments: Passed to the splits method of
                # Dataset.
        # """
        # return super(SST_LST, cls).splits(
            # root=root, text_field=text_field, path=root, label_field=label_field,
            # train=train, validation=None, test=test, **kwargs)



import numpy as np
from torchtext import data
class SST(data.Dataset):

    urls = ['']
    name = 'sst_data'
    dirname = 'data'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset instance given a path and fields.

        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        temp_data = np.load(path,allow_pickle=True)
        x_train = temp_data['train']
        y_train = temp_data['label']
        for i in range(len(x_train)):
            examples.append(data.Example.fromlist([x_train[i], 'pos' if y_train[i] == 1 else 'neg'], fields))
            # examples.append(data.Example.fromlist([x_train[i], y_train[i]], fields))
        super(SST, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='toxic_train.npz', test='toxic_test.npz', **kwargs):
        """Create dataset objects for splits of the IMDB dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(SST, cls).splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)



class SST_LST(data.Dataset):


    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset instance given a path and fields.

        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        # path = npy/train npy/test
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        # print(path)
        temp_data = kwargs[path]

        #
        # x_train = temp_data['train']
        # y_train = temp_data['label']
        for item in temp_data:

            # examples.append(data.Example.fromlist([x_train[i], 'pos' if y_train[i] == 1 else 'neg'], fields))
            ex = data.Example()
            setattr(ex, 'text', item[1])
            setattr(ex, 'label', item[0])
            examples.append(ex)
        super(SST_LST, self).__init__(examples, fields)

    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='train', test='test', **kwargs):
        """Create dataset objects for splits of the IMDB dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(SST_LST, cls).splits(
            root=root, text_field=text_field, path=root, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)
