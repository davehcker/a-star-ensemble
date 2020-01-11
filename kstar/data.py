import  numpy as np

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
    lambda x: pad_trim(np.flip(x, axis=0), size=50)[:, 0:1].flatten(),
    lambda x: pad_trim(np.flip(x, axis=0), size=50)[:, 1:2].flatten(),
    lambda x: pad_trim(x, size=50)[:, 0:1].flatten(),
    lambda x: pad_trim(x, size=50)[:, 1:2].flatten(),
    lambda x: pad_trim(x[:5], size=50)[:, 0:2].flatten(),
    lambda x: pad_trim(np.abs(x - x[0]))[:, 0:2].flatten(),
    lambda x: pad_trim(np.abs(np.multiply(x - x[0], np.log(x[:, 2:]))))[:, 0:2].flatten(),
    # PCA
    #    lambda x: pad_trim(x, size=meanMouseSessionLength)[:, 0:2].flatten(),
]
