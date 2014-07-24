from decision_tree import DecisionTree
from data import Instance
from algo import Id3
from decision_tree_refiner import DecisionTreeRefiner
import StringIO


def training(train_filepath):
    train_data = Instance.read(train_filepath)

    algo = Id3()
    dt = DecisionTree(train_data, algo)
    path = dt.train()
    return path

def validation(valid_filepath, path):
    validation_data = Instance.read(valid_filepath)

    r = DecisionTreeRefiner()
    result = r.refine(path, validation_data)

    return result

def create_query(dtr):
    return ' OR '.join(['( %s )' % _create_and_queries(p) for p in dtr.paths])

def _create_and_queries(path):
    return ' '.join([nd[0] if nd[1] else '-%s' % nd[0] for nd in path.node_decisions])

if __name__ == '__main__':
    import glob
    import time
    import optparse
    import os
    parser = optparse.OptionParser(usage="usage: %prog [options] train_filepath valid_filepath")
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("needs train_filepath and valid_filepath")
    train_filepath = args[0]
    valid_filepath = args[1]

    path = training(train_filepath)
    result = validation(valid_filepath, path)
    print create_query(result)

