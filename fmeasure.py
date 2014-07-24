import numpy as np
class Fmeasure:
    def __init__(self, relnum, beta = 1.0):
        self.relnum = relnum
        self.beta = beta

    def gain(self, node):
        # original f
        original_f = self._f(node.instances)

        # prepare
        data_dict = {}
        for d in node.instances:
            data_dict[d.id] = d

        gains = []
        for w in node.dictionary:
            pos = [data_dict[d] for d in node.index[w] if d in data_dict]
            neg = [data_dict[d] for d in data_dict.keys() if not d in node.index[w]]
            if len(pos) == 0 or len(neg) == 0:
                max_f = 0.0
            else:
                max_f = self._max_f([pos, neg])
            max_gain = max_f - original_f
            gains.append((w, max_gain))

        return gains

    def _max_f(self, datas):
        result = np.max([self._f(data) for data in datas])
        return result

    def _f(self, data):
        if len(data) == 0:
            return 0
        tdata = [d for d in data if d.label == 1]
        precision = float(len(tdata)) / len(data)
        recall = float(len(tdata)) / self.relnum
        if precision == 0 or recall == 0:
            return 0.0
        else:
            result = (1.0 + self.beta ** 2) / ((self.beta ** 2 / recall) + (1.0 / precision))
            return result
