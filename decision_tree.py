import numpy as np

# Decision Tree
# e.g.
# instances = Instance.load('something.dat')
# algorithm = Id3() # algorithm that partitions instances
# tree = DecisionTree(instances, algorithm)
# res = tree.train()
class DecisionTree:
    def __init__(self, instances, algorithm):
        self.instances = instances
        self.algorithm = algorithm
        dictionary = self._create_dictionary(instances)
        index = self._create_inverted_index(instances)
        # create a root node
        self.root = Node(instances, algorithm, dictionary, index)

    # Train a decision tree to output paths
    def train(self):

        # grow until false returned
        is_working = True
        while is_working:
            is_working = self.root.grow()

        # output DecisionTreeResult
        paths = self.root.path([])
        result = DecisionTreeResult(paths)
        return result

    # List up all words
    def _create_dictionary(self, instances):
        result = set()
        for d in instances:
            result = result.union(d.features)
        return result

    # Create an inverted index
    def _create_inverted_index(self, instances):
        result = {}
        for d in instances:
            for f in d.features:
                if not f in result:
                    result[f] = set()
                result[f].add(d.id)
        return result

# Internal node for decision tree
class Node:
    def __init__(self, instances, algorithm, dictionary, index):
        self.instances = instances
        self.algorithm = algorithm
        self.dictionary = dictionary
        self.index = index
        self.working = True
        self.feature = None
        self.pos = None
        self.neg = None
        self.label = None

    # Return paths from the root to leaves
    def path(self, prev):
        if self.pos is None and self.neg is None:
            return [Path(prev, self.label)]
        else:
            pos_prev = list(prev)
            pos_prev.append((self.feature, True))
            pos = self.pos.path(pos_prev)
            neg_prev = list(prev)
            neg_prev.append((self.feature, False))
            neg = self.neg.path(neg_prev)
            return pos + neg

    # Try to grow. Return True if it grows; otherwise False.
    def grow(self):
        if self.working:
            self._split()
            return True
        else:
            if self.pos is not None and self.neg is not None:
                return self.pos.grow() or self.neg.grow()
            else:
                return False

    # Split instances with the feature that maximizes the gain
    def _split(self):
        # finish working
        self.working = False
        # terminate if labeled
        if self._try_to_assign_label():
            return False
        # gain computation
        gains = self.algorithm.gain(self)
        # find max
        feature, gain = self._argmax_gains(gains)
        # return False if it cannot grow
        if gain <= 0:
            return False
        # debug
        print "Selected: %s, Gain: %s" % (feature, gain)
        # create children and return True
        self._create_children(feature)
        return True

    # Create children
    def _create_children(self, feature):
        # split instances
        pos = [d for d in self.instances if feature in d.features]
        neg = [d for d in self.instances if not feature in d.features]
        # remove a used feature
        new_dictionary = set(self.dictionary)
        new_dictionary.remove(feature)
        # set feature, pos, and neg
        self.feature = feature
        self.pos = Node(pos, self.algorithm, new_dictionary, self.index)
        self.neg = Node(neg, self.algorithm, new_dictionary, self.index)

    # Assign a label if instances are homogeneous
    def _try_to_assign_label(self):
        for i in [0, 1]:
            if all([d.label == i for d in self.instances]):
                self.label = i
                return True
        return False

    # Find the feature that maximize the gain
    def _argmax_gains(self, gains):
        max_index = np.argmax([g[1] for g in gains])
        feature = gains[max_index][0]
        max_gain = gains[max_index][1]
        return (feature, max_gain)


# Decision tree result that contains only Paths
class DecisionTreeResult:
    def __init__(self, paths):
        self.paths = paths

    def dump(self, filepath):
        import pickle
        with open(filepath, 'w') as f:
            pickle.dump(self, f)

    def query(self, instances):
        return [d for d in instances if self.match(d)]

    def is_subset(self, other):
        if type(other) is type(self):
            return all([any([sp == op for op in other.paths]) for sp in self.paths])

    def match(self, d):
        return any([p.match(d) for p in self.paths])

    def __eq__(self, other):
        return self.is_subset(other) and other.is_subset(self)

    def __str__(self):
        return '\n'.join(map(str, self.paths))

    @classmethod
    def load(cls, filepath):
        import pickle
        with open(filepath, 'r') as f:
            result = pickle.load(f)
        return result

# Path consists of a list of node_decisions (a pair of a feature and T/F) and leaf label
class Path:
    def __init__(self, node_decisions, label):
        self.node_decisions = node_decisions
        self.label = label

    def query(self, instances):
        if len(self.node_decisions) == 0:
            return instances
        return [d for d in instances if self.match(d)]

    def is_subset(self, other):
        if type(other) is type(self):
            return (self.label == other.label 
                and all([any([snd[0] == ond[0] and snd[1] == ond[1] for ond in other.node_decisions]) 
                for snd in self.node_decisions]))

    def match(self, d):
        return all([self._node_match(n, d) for n in self.node_decisions])

    def _node_match(self, node, d):
        return (node[1] == True and node[0] in d.features
                or node[1] == False and not node[0] in d.features)

    def __eq__(self, other):
        return self.is_subset(other) and other.is_subset(self)
    
    def __str__(self):
        return '%s:%s' % (self.label, 
            '^'.join(['%s' % nd[0] if nd[1] ==1 else '-%s' % nd[0] for nd in self.node_decisions]))


if __name__ == '__main__':
    from data import Instance
    from algo import Id3, Gini
    from fmeasure import Fmeasure
    import time
    import optparse
    import os
    parser = optparse.OptionParser(usage="usage: %prog [options] filepath")
    parser.add_option("-a", type="choice", choices=['id3', 'gini', 'f'],
        dest="algo", help="algorithm for decision tree", default="id3")
    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.error("needs filepath")

    start_time = time.clock()
    filepath = args[0]
    instances = Instance.read(filepath)
    print '%s used (#pos: %s, #neg: %s)' % (filepath, 
        len([d for d in instances if d.label == 1]),
        len([d for d in instances if d.label != 1]))

    if options.algo == 'id3':
        algo = Id3()
    elif options.algo == 'gini':
        algo = Gini()
    elif options.algo == 'f':
        relnum = len([d for d in instances if d.label == 1])
        algo = Fmeasure(relnum)

    dt = DecisionTree(instances, algo)
    dpath = dt.train()
    print "Paths generated:"
    for p in dpath.paths:
        print p
    dpath.dump('%s.path' % filepath)

    end_time = time.clock()
    print 'Time:', end_time - start_time

