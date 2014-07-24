
# script for browsing the decision tree
if __name__ == '__main__':
    from data import Instance
    from decision_tree import DecisionTreeResult, Path
    import optparse
    import os
    parser = optparse.OptionParser(usage="usage: %prog [options] filepath")
    (options, args) = parser.parse_args()
    if len(args) < 1:
        parser.error("needs filepath")

    filepath = args[0]
    dtresult = DecisionTreeResult.load(filepath)
    for p in dtresult.paths:
        print p
