from contextlib import contextmanager
import copy
import math
import networkx as nx
import random
from scipy.stats import binom

MAX_VALUE = 10
NUM_FUNCTION_INSTANCES = 3
NUM_ITEMS = 3


class FunctionInstance(object):

    def __init__(self, idx):
        self._data = set()
        self.idx = idx

    def depends_on(self, data_item):
        """
        mark this function instance as dependent on *data_item*
        """
        self._data.add(data_item)
        data_item.dependent_fi.add(self)

    @property
    def data(self):
        """
        immutable version of the data (and thus hashable)
        """
        return frozenset(self._data)


class DataItem(object):

    def __init__(self, idx, value):
        self.idx = idx
        self.value = value

        # function instances dependent on this data item
        self.dependent_fi = set()

    def __repr__(self):
        return "item(idx={}, value={})".format(
            repr(self.idx),
            repr(self.value),
        )

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

    @property
    def possible_changes(self):
        """
        all possible values, excluding the current one
        """
        if not hasattr(self, '_possible_changes'):
            self._possible_changes = self.possible_values()
            self._possible_changes.remove(self.value)
        return self._possible_changes


def joint_probability(constants=None, changes=None):
    """
    joint probability of this dataset (with the current values) occurring
    assume data change events are independent
    """
    prob = 1

    # constant items remain unchanged (thus keeping their old probability)
    if constants is not None:
        for constant in constants:
            prob *= constant.probability(constant.value)

    # change the changed items to the changed_value
    if changes is not None:
        for change in changes:
            prob *= change.probability(change.changed_value)

    return prob


def importance(changes=None):
    """
    importance is given by the total number of dependent function instances
    of the associated changing items
    """
    dependent_fis = set()
    if changes is not None:
        for change in changes:
            dependent_fis |= change.dependent_fi
    return len(dependent_fis)


def weight(constants=None, changes=None):
    """
    weight of this changeset taking the current values
    """
    return (
        importance(changes) * joint_probability(constants, changes)
    )


class ChangeableDataItem(DataItem):

    def __init__(self, idx, value):
        super(ChangeableDataItem, self).__init__(idx, value)

        self.change_idx = 0

    def __repr__(self):
        return "change(" + super(ChangeableDataItem, self).__repr__() \
            + " ==> " + str(self.changed_value) + ")"

    @property
    def changed_value(self):
        """
        what is the current value of the associated item?
        """
        return self.possible_changes[self.change_idx]

    @property
    def prob_change(self):
        """
        what is the probability of the item taking it's current changed_value?
        """
        return self.probability(self.changed_value)

    @property
    def prob_const(self):
        """
        what is the probability of the item taking it's current changed_value?
        """
        return self.probability(self.value)

    @contextmanager
    def fake_change(self):
        """
        increase, yield, decrease change_idx
        """
        self.change_idx += 1
        yield
        self.change_idx -= 1

    def can_change(self):
        """
        whether this item has another change to assess
        """
        return self.change_idx < len(self.possible_changes) - 1

    def change(self):
        """
        go ahead and perform the change
        """
        self.change_idx += 1


class ChangeableDataset(object):

    def __init__(self, fis, items):

        # sanity checks: sets are disjoint and contain data items
        for item in items:
            assert isinstance(item, ChangeableDataItem)

        self.fis = fis
        self.ordered_fis = sorted(list(fis), key=lambda fi: fi.idx)

        self.items = items
        self.ordered_items = sorted(list(items), key=lambda item: item.idx)

    def __repr__(self):
        return "ChangeableDataset(items={})".format(repr(self.items))

    def change(self, items):
        """
        change this dataset to its next best possible values by mutating
        one of the items specified
        return "false" when no changes left to make
        """
        items = list(items)
        probabilities = []

        for item in items:
            assert item in self.items
            if item.can_change():

                # fake the change (incrementing the change_idx)
                with item.fake_change():

                    # assess the joint probability of changing to new values
                    probabilities.append(
                        joint_probability(changes=items)
                    )

            else:
                probabilities.append(0)

        assert len(probabilities) == len(items)

        if sum(probabilities) > 0:
            items[probabilities.index(max(probabilities))].change()
            return probabilities
        return False

    def construct_graph(self):
        """
        construct the corresponding nx.DiGraph()
        """
        graph = nx.DiGraph()

        # data items connected to sink with weight abs(ln(prob_const))
        for item in self.items:
            capacity = abs(math.log(item.prob_const))
            graph.add_edge('d' + str(item.idx), 't', capacity=capacity)

        # source connected to fis with prob = min(abs(ln(prob_change)))
        for fi in self.fis:
            capacity = min(
                abs(math.log(item.prob_change))
                for item in fi.data
            )
            graph.add_edge('s', 'f' + str(fi.idx), capacity=capacity)

            # connect fi to dependent data with effectively "infinite" weight
            for item in self.items:
                graph.add_edge(
                    'f' + str(fi.idx),
                    'd' + str(item.idx),
                    capacity=100,
                )

        return graph

    def cut(self):
        """
        perform min-cut and compute constants / changing data items
        """
        graph = self.construct_graph()

        cut_value, partition = nx.minimum_cut(graph, 's', 't')
        reachable, non_reachable = partition

        # keep constant all reachable data items
        constants = set()
        for label in reachable:
            if label[0] == 'd':
                constants.add(self.ordered_items[int(label[1:])])

        # the rest should change
        changes = set(self.items) - constants
        return constants, changes

    def generate(self):
        """
        enumerate all possible permutations of changes on the data items
        """

        # while at least one data item can change
        while sum(int(item.can_change()) for item in self.items) > 0:

            constants, changes = self.cut()
            dataset = FrozenDataset(constants, changes)
            if dataset.weight() > 0:
                yield dataset
            self.change(changes)


class FrozenDataset(object):

    def __init__(self, constants, changes):
        self.constants = copy.deepcopy(constants)
        self.changes = copy.deepcopy(changes)

    def __repr__(self):
        return "FrozenDataset(constants={}, changes={})".format(
            repr(self.constants),
            repr(self.changes),
        )

    def joint_probability(self):
        return joint_probability(self.constants, self.changes)

    def importance(self):
        return importance(self.changes)

    def weight(self):
        return weight(self.constants, self.changes)


def construct_sample():
    # construct the function instances
    fis = [FunctionInstance(i) for i in range(NUM_FUNCTION_INSTANCES)]
    items = []

    # construct the data items
    for i in range(NUM_ITEMS):
        d = ChangeableDataItem(i, random.randrange(MAX_VALUE))

        # depend on function instances [0, i)
        for f in range(i + 1):
            fis[f].depends_on(d)

        items.append(d)

    return frozenset(fis), frozenset(items)


if __name__ == '__main__':
    fis, items = construct_sample()

    old_weight = -1
    obj = ChangeableDataset(fis, items)
    for changeset in obj.generate():
        print(
            '{}: joint_probability={:.2E}, importance={}, '
            'weight={:.2E}'.format(
                repr(changeset),
                changeset.joint_probability(),
                changeset.importance(),
                changeset.weight(),
            )
        )
        if old_weight > 0:
            assert changeset.weight() <= old_weight
        old_weight = changeset.weight()
