import numpy as np
from torchtext import data
class TOXIC(data.Dataset):

    urls = ['']
    name = 'toxic_data'
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
        super(TOXIC, self).__init__(examples, fields, **kwargs)

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
        return super(TOXIC, cls).splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)



class TOXIC_LST(data.Dataset):


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
        super(TOXIC_LST, self).__init__(examples, fields)

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
        return super(TOXIC_LST, cls).splits(
            root=root, text_field=text_field, path=root, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)
