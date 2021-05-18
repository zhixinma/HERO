import os
from matplotlib import pyplot as plt
import torch
import random
from collections import Counter
from collections.abc import Iterable   # import directly from collections for Python < 3.3
import numpy as np
from typing import Union
import networkx as nx


# List operation
def deduplicate(arr: list):
    """
    :param arr: the original list
    :return: deduplicated list
    """
    return list(set(arr))


def is_overlap(a, b):
    """
    :param a: list to compare
    :param b: list to compare
    :return: True if there are common element in a and b
    """
    a, b = deduplicate(a), deduplicate(b)
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    for e in shorter:
        if e in longer:
            return True
    return False


def subtract(tbs, ts):
    """
    :param tbs: list to be subtracted
    :param ts: list to subtract
    :return: subtracted list a
    """
    ts = deduplicate(ts)
    return [e for e in tbs if e not in ts]


def expand_window(data, window_set=1):
    """
    :param data: the original list
    :param window_set: int or list of int
    :return: a list of lists shifted by the bias which is specified by window_set
    """
    if isinstance(window_set, int):
        window_set = [window_set]
    window_list = []
    for bias in window_set:
        if bias > 0:
            tmp = [data[0]] * bias + data[:-bias]
        elif bias < 0:
            tmp = data[-bias:] + [data[-1]] * -bias
        else:
            tmp = data
        window_list.append(tmp)
    return window_list


def mean(data: list):
    """
    :param data: a list of int
    :return: the average of integers in data
    """
    res = sum(data) / len(data)
    return res


def flatten(list_of_lists):
    """
    :param list_of_lists: a list of sub-lists like "[[e_{00}, ...], ..., [e_{n0}, ...]]"
    :return: flatten all sub-lists into single list
    """
    return [item for sublist in list_of_lists for item in sublist]


# Numpy operation
def pairwise_equation(data: np.ndarray, tok_illegal=None):
    """
    :param data: the original data array
    :param tok_illegal: the tokens which is meaningless
    :return: an indicator matrix A where A_{i, j} == 1 denotes data[i] == data[j]
    """
    placeholder = "1&*^%!2)!"  # an impossible string
    str_mat = data.reshape(len(data), 1).repeat(len(data), axis=1)
    if tok_illegal is not None:
        data[data == tok_illegal] = placeholder
    mat = str_mat == data
    indicator_matrix = np.tril(mat, -1).astype(int)
    return indicator_matrix


def indicator_vec(indices: Union[int, list], n: int, device=None, dtype=torch.float):
    """
    :param indices: indices which is one
    :param n: number of classes
    :param device:
    :param dtype:
    :return: an indicator vector
    """
    vec = torch.zeros(n, dtype=dtype)
    vec[indices] = 1
    if device is not None:
        vec = vec.to(device)
    return vec


# Dict operation
def sample_dict(d: dict, n: int, seed=None):
    """
    :param d: original dict
    :param n: number of keys to sample
    :param seed: random seed of sampling
    :return: sampled dictionary
    """
    if seed is not None:
        random.seed(seed)
    keys = random.sample(d.keys(), n)
    sample_d = {k: d[k] for k in keys}
    return sample_d


# Tensor operation
def cosine_sim(fea: torch.Tensor):
    """
    :param fea: feature vector [N, D]
    :return: score: cosine similarity score [N, N]
    """
    fea /= torch.norm(fea, dim=-1, keepdim=True)
    score = fea.mm(fea.T)
    return score


def uniform_normalize(t: torch.Tensor):
    """
    :param t:
    :return: normalized tensor
    >>> a = torch.rand(5)
    tensor([0.3357, 0.9217, 0.0937, 0.1567, 0.9447])
    >>> uniform_normalize(a)
    tensor([0.2843, 0.9730, 0.0000, 0.0740, 1.0000])
    """
    t -= t.min(-1, keepdim=True)[0]
    t /= t.max(-1, keepdim=True)[0]
    return t


def build_sparse_adjacent_matrix(edges: list, n: int, device=None, dtype=torch.float, undirectional=True):
    """
    Return adjacency matrix
    :param edges: list of edges, for example (st, ed)
    :param n: number of vertices
    :param device:
    :param dtype:
    :param undirectional: make adjacency matrix un-directional
    :return: the sparse adjacent matrix
    """
    i = torch.tensor(list(zip(*edges)))
    v = torch.ones(i.shape[1], dtype=dtype)
    sparse = torch.sparse_coo_tensor(i, v, (n, n))
    if device is not None:
        sparse = sparse.to(device)
    a = sparse.to_dense()
    if undirectional:
        ud_a = ((a > 0) | (a.transpose(-2, -1) > 0)).to(dtype)
        a = ud_a
    return a


# List statistic
def percentile(data: list, p=0.5):
    """
    :param data: origin list
    :param p: frequency percentile
    :return: the element at frequency percentile p
    """
    assert 0 < p < 1
    boundary = len(data) * p
    counter = sorted(Counter(data).items(), key=lambda x: x[0])
    keys, counts = zip(*counter)
    accumulation = 0
    for i, c in enumerate(counts):
        accumulation += c
        if accumulation > boundary:
            return keys[i]
    return None


# List visualization
def save_plt(fig_name: str, mute):
    """
    :param fig_name: path of target file
    :param mute: mute the output if True
    :return: None
    """
    fig_dir = fig_name[:fig_name.rfind("/")]
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_name)
    if not mute:
        print("'%s' saved." % fig_name)


def list_histogram(data: list, color="b", title="Histogram of element frequency.", x_label="", y_label="Frequency", fig_name="hist.png", mute=False):
    """
    :param data: the origin list
    :param color: color of the histogram bars
    :param title: bottom title of the histogram
    :param x_label: label of x axis
    :param y_label: label of y axis
    :param fig_name: path of target file
    :param mute: mute the output if True
    :return: None
    """
    def adaptive_bins(_data):
        """
        :param _data: the original list to visualize
        :return: the adaptive number of bins
        """
        n = len(deduplicate(_data))
        return n

    bins = adaptive_bins(data)
    plt.hist(data, color=color, bins=bins)
    plt.gca().set(xlabel=x_label, ylabel=y_label)
    plt.title("Fig. "+title, fontdict={'family': 'serif', "verticalalignment": "bottom"})
    save_plt(fig_name, mute)
    plt.clf()


def show_type_tree(data, indentation=4, depth=0, no_leaf=True):
    """
    :param data: the data to show the structure
    :param indentation: number of space of indentation
    :param depth: variable used for recursive
    :param no_leaf: don't display the leaf (non-iterable) node if True
    :return: None
    """
    def _indent(content: str):
        if depth == 0:
            print()
        print(" " * (depth * indentation) + content)

    if not isinstance(data, Iterable):
        if no_leaf:
            return

        if isinstance(data, int):
            _indent("int: %d" % data)
        elif isinstance(data, float):
            _indent("float: %.2f" % data)
        else:
            _indent(str(type(data)))
        return

    if isinstance(data, list):
        _indent("list with size %d" % len(data))
        for item in data:
            show_type_tree(item, depth=depth + 1)

    elif isinstance(data, tuple):
        _indent("tuple with size %d" % len(data))
        for item in data:
            show_type_tree(item, depth=depth + 1)

    elif isinstance(data, dict):
        _indent("dict with size %d" % len(data))
        for key in data:
            _indent(str(key))
            show_type_tree(data[key], depth=depth + 1)

    elif isinstance(data, str):
        _indent("str: " + data)

    elif isinstance(data, torch.Tensor):
        _indent("Tensor with shape" + str(list(data.shape)))

    else:
        _indent(str(type(data)))
        for item in data:
            show_type_tree(item, depth=depth + 1)


def random_color(seed=None):
    if seed:
        random.seed(seed)  # cover all 6 times of sampling
    color = "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
    return color
