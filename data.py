class Instance:
    id = None
    label = None
    features = None

    @classmethod
    def parse(cls, line):
        ls = [l.strip() for l in line.split('\t')]
        i = Instance()
        i.id = ls[0]
        i.label = int(ls[1])
        i.features = set([f for f in ls[2].split(' ') if len(f) > 0])
        return i

    @classmethod
    def read(cls, filepath):
        result = []
        with open(filepath, 'r') as f:
            for line in f:
                i = Instance.parse(line)
                result.append(i)
        return result

    def __str__(self):
        return '%s(%s): %s' % (self.id, self.label, ', '.join(list(self.features)))


if __name__ == '__main__':
    instances = Instance.read('./dt/0.dat')
    for i in instances:
        print i
