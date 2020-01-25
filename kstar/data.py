import  numpy as np
import random
from sklearn.model_selection import train_test_split


def pad_trim(data, size=100, padding_method='zeros'):
    """
    max_len: the size of the final list
    padding_method:
        'zeros': adds zeros (0) for missing numeric values
        'last': duplicates last value
        'interpolated': interpolatest the values
    """

    if len(data) == 0:
        pass

    elif len(data) > size:
        data = data[:size]

    else:
        if padding_method == 'zeros':
            padding = np.tile(np.array([0] * len(data[0])), reps=(size - len(data), 1))
            data = np.vstack([data, padding])

    return data


def pick_n(data, n):
    """
    pick only n-th rows
    """
    return data[::n]


data_transformations = [
    lambda x: pad_trim(np.flip(x, axis=0), size=self.mean_length)[:, 0:1].flatten(),
    lambda x: pad_trim(np.flip(x, axis=0), size=self.mean_length)[:, 1:2].flatten(),
    lambda x: pad_trim(x, size=50)[:, 0:1].flatten(),
    lambda x: pad_trim(x, size=50)[:, 1:2].flatten(),
    lambda x: pad_trim(x[:5], size=50)[:, 0:2].flatten(),
    lambda x: pad_trim(np.abs(x - x[0]))[:, 0:2].flatten(),
    lambda x: pad_trim(np.abs(np.multiply(x - x[0], np.log(x[:, 2:]))))[:, 0:2].flatten(),
    # PCA
    #    lambda x: pad_trim(x, size=meanMouseSessionLength)[:, 0:2].flatten(),
]

keystroke_data_transformation = [
        lambda x: np.histogram(x, bins=50)[0].flatten()
]

def calculate_dataset_size(data, user, source_ratio=(1.0, 1.0)):
    """
    Gives the total number of sessions that belong to <user> and those
    that do not.
    """
    source_ratio_self, source_ratio_other = source_ratio

    source_len = {_:0 for _ in set(map(lambda x: x[-1], data))}
    for item in data: source_len[item[-1]] += 1

    if type(source_ratio_self) == float:
        source_ratio_self = int(source_len[user] * source_ratio[0])
    if type(source_ratio_other) == float:
        source_ratio_other = int(sum(list(map(lambda x: x[-1], filter(lambda x: x[0] != user, source_len.items())))) * source_ratio[1])

    source_ratio = (source_ratio_self, source_ratio_other)
    return source_ratio


def generate_train_test(data, user, source_ratio=(1.0, 0.5), test_size=0.2):
    source_count = calculate_dataset_size(data, user, source_ratio)
    trainX, testX, trainY, testY = None, None, None, None

    user_sessions = list(filter(lambda x: x[1] == user, data))
    non_user_sessions = list(filter(lambda x: x[1] != user, data))

    trainX, testX, trainY, testY = train_test_split(user_sessions, [1] * len(user_sessions), test_size=test_size)

    non_user_samples = random.sample(non_user_sessions, source_count[-1])
    non_user_train_test_samples = train_test_split(non_user_samples, [0] * len(non_user_samples), test_size=test_size)

    trainX += non_user_train_test_samples[0]
    testX += non_user_train_test_samples[1]
    trainY += non_user_train_test_samples[2]
    testY += non_user_train_test_samples[3]

    return trainX, testX, trainY, testY

