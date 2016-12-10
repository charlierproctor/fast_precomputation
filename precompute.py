import random
from scipy.stats import binom

MAX_VALUE = 10
NUM_FUNCTION_INSTANCES = 3
NUM_ITEMS = 3


class FunctionInstance(object):

    def __init__(self):
        self.data = set()

    def depends_on(self, data_item):
        """
        mark this function instance as dependent on *data_item*
        """
        self.data.add(data_item)
        data_item.dependent_fi.add(self)


class DataItem(object):

    def __init__(self, idx, value):
        self.idx = idx
        self.value = value

        # function instances dependent on this data item
        self.dependent_fi = set()

    def __repr__(self):
        return self.idx.__repr__()

    def probability(self, value):
        """
        probability that this data item takes on the given value
        assume it's Binomial(MAX_VALUE, 0.5)
        """
        return binom.pmf(value, n=MAX_VALUE, p=0.5)

    def possible_values(self):
        """
        return possible values, sorted by decreasing probability
        assume this item can take values from [0, MAX_VALUE)
        """
        return sorted(range(MAX_VALUE), key=self.probability, reverse=True)

    def possible_changes(self):
        """
        all possible values, excluding the current one
        """
        possible_values = self.possible_values()
        possible_values.remove(self.value)
        return possible_values


def joint_probability(changed_items, changed_values):
    """
    joint probability of a set of data changes occuring
    assume data change events are independent
    """
    prob = 1
    for item, value in zip(changed_items, changed_values):
        prob *= item.probability(value)
    return prob


def importance(changed_items):
    """
    importance is given by the total number of dependent function instances
    """
    dependent_fis = set()
    for item in changed_items:
        dependent_fis |= item.dependent_fi
    return len(dependent_fis)


def weight(items, values):
    changed = [
        (item, value)
        for item, value in zip(items, values)
        if item.value != value
    ]
    if changed:
        changed_items, change_values = zip(*changed)

        return (
            importance(changed_items)
            * joint_probability(changed_items, change_values)
        )
    else:
        return 0


def important_datasets(fis, skip=set()):
    """
    Yield datasets in decreasing order of importance,
    where *importance* is the number of dependent function instances
    skip: datasets to skip (not yield)
    verify_important_datasets verifies the properties of this function
    """

    def important_datasets_generator(fis):
        if len(fis) == 0:
            yield frozenset()
        else:
            # recursively generate datasets (sets of data items to change)
            for dataset in important_datasets(fis[1:]):

                # change data this function instance depends on
                for dependent_data_item in fis[0].data:
                    yield frozenset(set({dependent_data_item}) | dataset)

                # exclude this function instance
                yield dataset

    for dataset in important_datasets_generator(fis):
        if dataset not in skip:
            skip.add(dataset)
            yield dataset


def verify_important_datasets(important):
    """
    verify properties of dataset list returned by important_datasets()
    """
    assert len(set(important)) == len(important), "datasets should be unique"
    previous_importance = None
    for dataset in important:
        dataset_importance = importance(dataset)
        if previous_importance is None:
            previous_importance = dataset_importance
        else:
            assert dataset_importance <= previous_importance, \
                "importance must decrease"


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
    # construct the function instances
    fis = [FunctionInstance() for _ in range(NUM_FUNCTION_INSTANCES)]
    items = []

    # construct the data items
    for i in range(NUM_ITEMS):
        d = DataItem(i, random.randrange(MAX_VALUE))

        # depend on function instances [0, i)
        for f in range(i + 1):
            fis[f].depends_on(d)

        items.append(d)

    important = list(important_datasets(fis))
    verify_important_datasets(important)

    best = max(permute(items), key=lambda values: weight(items, values))
    print(best)


if __name__ == '__main__':
    main()
