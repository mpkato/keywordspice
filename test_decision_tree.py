import nose
from decision_tree import DecisionTree, DecisionTreeResult, Path
from data import Instance
from algo import Id3
from algo import Gini
from decision_tree import DecisionTree

class TestDecisionTree(object):
    def __init__(self):
        self.instances1 = Instance.read('./data/test1.dat')
        self.instances2 = Instance.read('./data/test2.dat')
        self.gt1 = DecisionTreeResult([
            Path([('a', True), ('b', True), ('c', True)], 0),
            Path([('a', True), ('b', True), ('c', False)], 1),
            Path([('a', True), ('b', False)], 0),
            Path([('a', False), ('c', True)], 1),
            Path([('a', False), ('c', False), ('q', True)], 1),
            Path([('a', False), ('c', False), ('q', False), ('e', True)], 1),
            Path([('a', False), ('c', False), ('q', False), ('e', False)], 0)
            ])
        self.gt2 = DecisionTreeResult([
            Path([('a', True), ('b', True), ('c', True)], 0),
            Path([('a', True), ('b', True), ('c', False)], 1),
            Path([('a', True), ('b', False)], 0),
            Path([('a', False), ('c', True)], 1),
            Path([('a', False), ('c', False), ('e', True)], 1),
            Path([('a', False), ('c', False), ('e', False)], 0)
            ])

    def test_train_with_id3(self):
        algo = Id3()
        dt1 = DecisionTree(self.instances1, algo)
        path1 = dt1.train()
        assert path1 == self.gt1

        dt2 = DecisionTree(self.instances2, algo)
        path2 = dt2.train()
        assert path2 == self.gt2

    def test_train_with_gini(self):
        algo = Gini()
        dt1 = DecisionTree(self.instances1, algo)
        path1 = dt1.train()
        assert path1 == self.gt1

        dt2 = DecisionTree(self.instances2, algo)
        path2 = dt2.train()
        assert path2 == self.gt2

if __name__ == '__main__':
    nose.main(argv=['nose', '-v'])
