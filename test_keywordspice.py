import nose
from keywordspice import training, validation, create_query

class TestGenerateQueries(object):
    def __init__(self):
        pass

    def test_generate_queries_with_test1(self):
        path = training('./data/test1.dat')
        dtresult = validation('./data/test1.dat', path)
        query = create_query(dtresult)
        assert query == '( b -c )'

    def test_generate_queries_with_test2(self):
        path = training('./data/test2.dat')
        dtresult = validation('./data/test2.dat', path)
        query = create_query(dtresult)
        assert query == '( -a ) OR ( -c e )'

if __name__ == '__main__':
    nose.main(argv=['nose', '-v'])
