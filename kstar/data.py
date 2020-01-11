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


flip = np.flip
