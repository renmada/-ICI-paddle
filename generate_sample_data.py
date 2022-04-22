import pickle
import os
import numpy as np
import random

random.seed(121)
np.random.seed(121)


def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


def save_data(obj, file):
    with open(file, 'wb+') as f:
        pickle.dump(obj, f, )


dataset_dir = './data/MiniImagenet/'
sample_data_dir = './data/sample_data/'
train_dir = os.path.join(dataset_dir, 'miniImageNet_category_split_train_phase_train.pickle')
val_dir = os.path.join(dataset_dir, 'miniImageNet_category_split_val.pickle')
test_dir = os.path.join(dataset_dir, 'miniImageNet_category_split_test.pickle')
os.makedirs(sample_data_dir, exist_ok=True)

train_data = load_data(train_dir)
dev_data = load_data(val_dir)
test_data = load_data(test_dir)

train_idx = np.random.permutation(len(train_data['labels']))[:1000]
dev_idx = np.random.permutation(len(dev_data['labels']))[:400]
test_idx = np.random.permutation(len(test_data['labels']))[:400]

train_data = {'catname2label': train_data['catname2label'],
              'labels': [train_data['labels'][x] for x in train_idx],
              'data': train_data['data'][train_idx]
              }

dev_data = {'catname2label': dev_data['catname2label'],
            'labels': [dev_data['labels'][x] for x in dev_idx],
            'data': dev_data['data'][dev_idx],
            'label2catname': dev_data['label2catname']
            }

test_data = {'catname2label': test_data['catname2label'],
             'labels': [test_data['labels'][x] for x in test_idx],
             'data': test_data['data'][test_idx],
             'label2catname': test_data['label2catname']
             }

# save_data(train_data, sample_data_dir + 'miniImageNet_category_split_train_phase_train.pickle')
save_data(dev_data, sample_data_dir + 'miniImageNet_category_split_val.pickle')
save_data(test_data, sample_data_dir + 'miniImageNet_category_split_test.pickle')
