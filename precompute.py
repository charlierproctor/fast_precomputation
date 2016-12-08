import random
from scipy.stats import binom

MAX_VALUE = 10
NUM_FUNCTION_INSTANCES = 10
NUM_ITEMS = 3


class DataItem(object):

    def __init__(self, idx, value):
        self.idx = idx
        self.value = value
        print('data item', idx, 'has value', value)

    def dependent_fi(self):
        """
        function instances dependent on this data item
        assume all fi's are dependent
        """
        return set(range(NUM_FUNCTION_INSTANCES))

    def possible_values(self):
        """
        assume this item can take values from [0, MAX_VALUE)
        """
        return range(MAX_VALUE)

    def probability(self, value):
        """
        probability that this data item takes on the given value
        assume it's Binomial(MAX_VALUE, 0.5)
        """
        return binom.pmf(value, n=MAX_VALUE, p=0.5)


def joint_probability(items, values):
    """
    joint probability of a set of data changes occuring
    assume data change events are independent
    """
    prob = 1
    for item, value in zip(items, values):
        prob *= item.probability(value)
    return prob


def importance(items, values):
    """
    importance is given by the total number of dependent function instances
    """
    dependent_fis = set()
    for item, value in zip(items, values):
        if item.value != value:
            dependent_fis |= item.dependent_fi()
    return len(dependent_fis)


def weight(items, values):
    return (
        importance(items, values) * joint_probability(items, values)
    )


def permute(items, start=[]):
    """
    enumerate all possible permutations of changes on the data items
    """
    if len(items) == 0:
        return [start]

    res = []
    for first_value in items[0].possible_values():
        res += permute(items[1:], start=start + [first_value])
    return res


def main():
    items = [
        DataItem(i, random.randrange(MAX_VALUE)) for i in range(NUM_ITEMS)
    ]

    best = max(permute(items), key=lambda values: weight(items, values))
    print(best)


if __name__ == '__main__':
    main()
