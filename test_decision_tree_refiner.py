import nose
from decision_tree import DecisionTree, DecisionTreeResult, Path
from data import Instance
from decision_tree_refiner import DecisionTreeRefiner
from algo import Id3

class TestDecisionTreeRefiner(object):
    def __init__(self):
        self.instances1 = Instance.read('./data/test1.dat')
        dt = DecisionTree(self.instances1, Id3())
        dpath = dt.train()
        dpath.dump('./data/test1.dat.path')
        self.path1 = DecisionTreeResult.load('./data/test1.dat.path')

        self.instances2 = Instance.read('./data/test2.dat')
        dt = DecisionTree(self.instances2, Id3())
        dpath = dt.train()
        dpath.dump('./data/test2.dat.path')
        self.path2 = DecisionTreeResult.load('./data/test2.dat.path')

        self.dtr = DecisionTreeRefiner()

    def test_conjunction_refine(self):
        crefined1 = self.dtr.conjunction_refine(self.path1, self.instances1)
        assert crefined1 == DecisionTreeResult([Path([('b', True), ('c', False)], 1)])
        crefined2 = self.dtr.conjunction_refine(self.path2, self.instances2)
        assert crefined2 == DecisionTreeResult([
            Path([('b', True), ('c', False)], 1),
            Path([('a', False)], 1),
            Path([('c', False), ('e', True)], 1)])

    def test_disjunction_refine(self):
        drefined1 = self.dtr.refine(self.path1, self.instances1)
        drefined2 = self.dtr.refine(self.path2, self.instances2)

if __name__ == '__main__':
    nose.main(argv=['nose', '-v'])
