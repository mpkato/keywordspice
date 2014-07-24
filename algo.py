import numpy as np

class Id3:
    def gain(self, node):
        # original entropy
        original = self._entropy(node.instances)

        # average entropy for each feature
        data_dict = {}
        for d in node.instances:
            data_dict[d.id] = d
        gains = []
        for w in node.dictionary:
            pos = [data_dict[d] for d in node.index[w] if d in data_dict]
            neg = [data_dict[d] for d in data_dict.keys() if not d in node.index[w]]
            avg_entropy = self._avg_entropy([pos, neg])
            gain = original - avg_entropy
            gains.append((w, gain))

        return gains
    
    def _avg_entropy(self, datas):
        result = 0
        total = sum([len(d) for d in datas])
        for data in datas:
            relent = float(len(data)) / total * self._entropy(data)
            result += relent
        return result

    def _entropy(self, data):
        tdata = [d for d in data if d.label == 1]
        fdata = [d for d in data if d.label != 1]
        dist = [len(tdata), len(fdata)]
        result = self._ent(dist)
        return result

    def _ent(self, sizes):
        result = 0
        total = sum(sizes)
        if any([size == 0 for size in sizes]):
            return 0.0
        for size in sizes:
            p = float(size) / total
            result += -p * np.log2(p)
        return result

class Gini:
    def gain(self, node):
        # original gini
        original = self._gini(node.instances)

        # average gini for each feature
        data_dict = {}
        for d in node.instances:
            data_dict[d.id] = d
        gains = []
        for w in node.dictionary:
            pos = [data_dict[d] for d in node.index[w] if d in data_dict]
            neg = [data_dict[d] for d in data_dict.keys() if not d in node.index[w]]
            avg_gini = self._avg_gini([pos, neg])
            gain = original - avg_gini
            gains.append((w, gain))

        return gains

    def _avg_gini(self, datas):
        result = 0
        total = sum([len(d) for d in datas])
        for data in datas:
            relent = float(len(data)) / total * self._gini(data)
            result += relent
        return result

    def _gini(self, data):
        tdata = [d for d in data if d.label == 1]
        fdata = [d for d in data if d.label != 1]
        dist = [len(tdata), len(fdata)]
        result = self._gini_compute(dist)
        return result

    def _gini_compute(self, sizes):
        result = 1
        total = sum(sizes)
        if any([size == 0 for size in sizes]):
            return 0.0
        for size in sizes:
            p = float(size) / total
            result -=  p ** 2
        return result
