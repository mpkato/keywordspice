import numpy as np
from decision_tree import DecisionTreeResult, Path

class DecisionTreeRefiner:
    
    def __init__(self, beta=1.0):
        self.beta = beta

    def refine(self, dtresult, instances):
        crefined = self.conjunction_refine(dtresult, instances)
        drefined = self.disjunction_refine(crefined, instances)
        return drefined

    # Refine a disjunction rule
    # Remove conjunction rules from the disjunction rule while F-measure improved
    def disjunction_refine(self, dtresult, instances):
        relnum = len([d for d in instances if d.label == 1])
        result = []
        cur = dtresult
        while cur is not None:
            res = self._disjunction_refine_path(cur, instances, relnum)
            if res is None:
                result = cur
            cur = res

        return result

    def _disjunction_refine_path(self, dtresult, instances, relnum):
        # stop if no path remains
        if len(dtresult.paths) == 0:
            return None

        # original f
        original_result = dtresult.query(instances)
        original_f = self._compute_f(original_result, relnum)

        # compute f of smaller disjunctions
        gains = self._compute_f_of_smaller_disjunctions(instances, relnum, dtresult)
        result, max_f = self._argmax_f(gains)
        # maximum gain
        max_gain = max_f - original_f

        if max_gain >= 0:
            # debug
            #print "%s remains (Gain: %s)" % (result, max_gain)
            return result
        else:
            # stop if maximum gain < 0
            return None

    def _compute_f_of_smaller_disjunctions(self, instances, relnum, dtresult):
        result = []
        for i in range(len(dtresult.paths)):
            paths = list(dtresult.paths)
            del paths[i]
            new_dtresult = DecisionTreeResult(paths)
            res = new_dtresult.query(instances)
            f = self._compute_f(res, relnum)
            result.append((new_dtresult, f))
        return result

    def _argmax_f(self, gains):
        gain_values = [g[1] for g in gains]
        max_index = np.argmax(gain_values)
        maximizer = gains[max_index][0]
        max_f = gains[max_index][1]
        return (maximizer, max_f)

    # Refine conjunction rules
    # Remove literals from each conjunction rule while F-measure improved
    def conjunction_refine(self, dtresult, instances):
        relnum = len([d for d in instances if d.label == 1])
        positive_paths = [p for p in dtresult.paths if p.label == 1]
        result = []
        for p in positive_paths:
            cur = p
            while cur is not None:
                res = self._conjunction_refine_path(cur, instances, relnum)
                if res is None:
                    result.append(cur)
                cur = res

        result = self._remove_duplication(result)
        result = self._remove_null_path(result)
        return DecisionTreeResult(result)

    def _remove_duplication(self, paths):
        result = []
        for path in paths:
            if not any([path == r for r in result]):
                result.append(path)
        return result

    def _remove_null_path(self, paths):
        result = []
        for path in paths:
            if len(path.node_decisions) > 0:
                result.append(path)
        return result

    def _conjunction_refine_path(self, path, instances, relnum):
        # stop if no node remains
        if len(path.node_decisions) == 0:
            return None

        # original f
        original_result = path.query(instances)
        original_f = self._compute_f(original_result, relnum)

        # compute f of smaller conjunctions
        gains = self._compute_f_of_smaller_conjunctions(instances, relnum, path)
        result, max_f = self._argmax_f(gains)
        # maximum gain
        max_gain = max_f - original_f

        if max_gain >= 0:
            # debug
            #print "%s remains (Gain: %s)" % (result, max_gain)
            return result
        else:
            # stop if maximu gain < 0
            return None

    def _compute_f_of_smaller_conjunctions(self, instances, relnum, path):
        result = []
        for i in range(len(path.node_decisions)):
            nds = list(path.node_decisions)
            del nds[i]
            new_path = Path(nds, path.label)
            res = new_path.query(instances)
            f = self._compute_f(res, relnum)
            result.append((new_path, f))
        return result

    def _compute_f(self, instances, relnum):
        if len(instances) == 0:
            return 0.0
        rel = len([d for d in instances if d.label == 1])
        precision = float(rel) / len(instances)
        recall = float(rel) / relnum
        if precision == 0 or recall == 0:
            return 0.0
        else:
            result = (1.0 + self.beta ** 2) / ((self.beta ** 2 / recall) + (1.0 / precision))
            return result


if __name__ == '__main__':
    from data import Instance
    import time
    import optparse
    import os
    parser = optparse.OptionParser(usage="usage: %prog [options] filepath")
    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.error("needs filepath")

    start_time = time.clock()
    filepath = args[0]
    instances = Instance.read(filepath)
    path = DecisionTreeResult.load('%s.path' % filepath)

    r = DecisionTreeRefiner()
    result = r.refine(path, instances)
    print "Refined paths:"
    for p in result.paths:
        print p
    result.dump('%s.path.refined' % filepath)

    end_time = time.clock()
    print end_time - start_time

