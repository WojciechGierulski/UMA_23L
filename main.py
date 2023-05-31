import copy

import numpy as np
import pandas as pd

from numpy import log2 as log


def list_remove(item, list):
    new_list = copy.deepcopy(list)
    new_list.remove(item)
    return new_list


class Node:
    all_classes = []
    all_attributes = []
    possible_values_attribute = {}

    def __init__(self, possible_attributes):
        self.childs = {}
        self.pred = None
        self.possible_attributes = possible_attributes
        self.total_counts = {class_: 0 for class_ in Node.all_classes}
        self.class_counts = {
            attr: {val: {class_: 0 for class_ in Node.all_classes} for val in self.possible_values_attribute[attr]}
            for attr in self.possible_attributes}
        self.attr = None


def get_subtable(df, attr, value):
    return df[df[attr] == value].reset_index(drop=True)


def update_counts(node, df):
    for i, row in df.iterrows():
        r_df = pd.DataFrame(row.to_frame().T.reset_index(drop=True))
        node.total_counts[r_df['class'].loc[0]] += 1
        for attr in r_df.keys()[:-1]:
            if attr in node.possible_attributes:
                node.class_counts[attr][r_df[attr].loc[0]][r_df['class'].loc[0]] += 1


def calc_entropies(node):
    results = {}
    for attr in node.class_counts.keys():
        results[attr] = calc_entropy(node, attr)
    return results


def calc_entropy(node, attr):
    e = 0
    summ = sum(node.total_counts.values())
    for val in node.class_counts[attr].keys():
        e += calc_entropy_val(node, attr, val) * (sum(node.class_counts[attr][val].values()) / summ)
    return e


def calc_entropy_val(node, attr, val):
    e = 0
    summ = sum(node.class_counts[attr][val].values())
    if summ == 0:
        return 0
    for count in node.class_counts[attr][val].values():
        if count != 0:
            e -= (count / summ) * log((count) / summ)
        else:
            e -= 0
    return e


def buildTree(df, tree):
    update_counts(tree, df)
    if sum(list(tree.total_counts.values())) == max(list(tree.total_counts.values())):  # check purity
        for k, v in tree.total_counts.items():
            if v == max(list(tree.total_counts.values())):
                val = k
        tree.pred = val
        return tree

    entropies = calc_entropies(tree)
    attr = min(list(entropies.keys()), key=lambda k: entropies[k])
    attValue = np.unique(df[attr])

    if tree.pred is not None:
        tree.attr = attr
        tree.pred = None
        for value in attValue:
            subtable = get_subtable(df, attr, value)
            tree.childs[value] = buildTree(subtable, Node(list_remove(tree.attr, tree.possible_attributes)))
        return tree

    if tree.attr != attr:
        tree.childs = {}
        tree.attr = attr
        for value in attValue:
            subtable = get_subtable(df, attr, value)
            tree.childs[value] = buildTree(subtable, Node(list_remove(tree.attr, tree.possible_attributes)))
        return tree

    for value in attValue:
        subtable = get_subtable(df, attr, value)
        if value not in tree.childs:
            tree.childs[value] = Node(list_remove(tree.attr, tree.possible_attributes))
        tree.childs[value] = buildTree(subtable, tree.childs[value])
    return tree


def predict_single(row, tree):
    if tree.pred is not None:
        return tree.pred
    attr = tree.attr
    if row[attr] not in tree.childs.keys():
        return -1
    return predict_single(row, tree.childs[row[attr]])


def predict(df, tree):
    y = []
    for i, row in df.iterrows():
        yy = predict_single(row, tree)
        y.append(yy)
    return y


if __name__ == '__main__':
    data = pd.read_csv("./data.csv")
    X = data.drop(columns=["class"], axis=1)
    Y = data["class"]

    Node.all_classes = list(Y.unique())
    Node.all_attributes = list(X.keys())
    for k in X.keys():
        Node.possible_values_attribute[k] = list(X[k].unique())

    tree = Node(Node.all_attributes)

    # train tree by batch
    # tree = buildTree(data, tree)

    # train tree by single instances
    for i, row in data.iterrows():
        r_df = pd.DataFrame(row.to_frame().T)
        tree = buildTree(r_df, tree)

    r = predict(data, tree)