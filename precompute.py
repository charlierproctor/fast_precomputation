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
        self._data.add(data_item.idx)
        data_item.dependent_fi.add(self.idx)

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

    def reset(self):
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
        if self.change_idx < 0:
            return 0
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
        return 0 <= self.change_idx < len(self.possible_changes) - 1

    def change(self):
        """
        go ahead and perform the change
        """
        self.change_idx += 1

    def terminate(self):
        """
        done changing this object
        note: the changed_value will now yield self.possible_changes[-1]
        """
        self.change_idx = -1

    @property
    def active(self):
        return self.change_idx >= 0


class ChangeableDataset(object):

    def __init__(self, fis, changes=None):

        changes = set() if changes is None else set(changes)

        # sanity checks: sets are disjoint and contain data items
        for item in changes:
            assert isinstance(item, ChangeableDataItem)

        # fis, changes are dicts mapping from idx -> obj
        self.fis = {fi.idx: fi for fi in fis}
        self.changes = {item.idx: item for item in changes}

    def __repr__(self):
        return "ChangeableDataset(items={})".format(repr(self.changes))

    def change(self, indices):
        """
        change this dataset to its next best possible values by mutating
        one of the items specified
        return "false" when no changes left to make
        """
        assert indices <= set(self.changes.keys())

        indices = list(indices)
        probabilities = []

        for idx in indices:
            item = self.changes[idx]
            if item.can_change():

                # fake the change (incrementing the change_idx)
                with item.fake_change():

                    # assess the joint probability of changing to new values
                    probabilities.append(
                        (
                            item,
                            joint_probability(changes=[
                                self.changes[idx] for idx in indices
                            ])
                        )
                    )

            else:
                item.terminate()
                probabilities.append((None, 0))

        assert len(probabilities) == len(indices)

        if sum(prob for _, prob in probabilities) > 0:
            best_item, _ = max(probabilities, key=lambda tup: tup[1])
            best_item.change()
            return True
        return False

    def construct_graph(self):
        """
        construct the corresponding nx.DiGraph()
        """
        graph = nx.DiGraph()

        # data items connected to sink with weight abs(ln(prob_const))
        for idx, item in self.changes.items():
            capacity = abs(math.log(item.prob_const))
            graph.add_edge('d' + str(idx), 't', capacity=capacity)

        # source connected to fis with prob = min(abs(ln(prob_change)))
        for idx, fi in self.fis.items():
            changeable_fi_deps = fi.data & set(self.changes.keys())
            if len(changeable_fi_deps) == 0:
                continue
            capacity = min(
                (
                    abs(math.log(self.changes[idx].prob_change))

                    # effectively infinite weight when inactive
                    if self.changes[idx].active else 100
                )
                for idx in changeable_fi_deps
            )
            graph.add_edge('s', 'f' + str(idx), capacity=capacity)

            # connect fi to dependent data with effectively "infinite" weight
            for item in changeable_fi_deps:
                graph.add_edge(
                    'f' + str(idx),
                    'd' + str(item),
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
        new_constants = set()
        for label in reachable:
            if label[0] == 'd':
                new_constants.add(int(label[1:]))

        # the rest should change
        new_changes = self.changes.keys() - new_constants
        return ({
            idx: self.changes[idx]
            for idx in new_constants
        }, {
            idx: self.changes[idx]
            for idx in new_changes
        })

    def generate(self):
        """
        enumerate all possible permutations of changes on the data items
        """

        already_generated = set()

        # while at least one data item can change
        while sum(int(i.can_change()) for i in self.changes.values()) > 0:

            new_constants, new_changes = self.cut()

            dataset = FrozenDataset(
                list(new_constants.values()),
                list(new_changes.values()),
            )

            # hash the dataset and remember that we've already yielded it
            ds_hsh = hash(dataset)
            if ds_hsh not in already_generated and dataset.weight() > 0:
                yield dataset
            already_generated.add(ds_hsh)

            # perform the change: when false, we're done!
            if not self.change(set(new_changes.keys())):
                return


class FrozenDataset(object):

    def __init__(self, constants, changes):
        self.constants = copy.deepcopy(constants)
        self.changes = copy.deepcopy(changes)

    def __repr__(self):
        return "FrozenDataset(constants={}, changes={})".format(
            repr(self.constants),
            repr(self.changes),
        )

    def __hash__(self):
        return hash((
            frozenset([
                (i.idx, i.value) for i in self.constants
            ]),
            frozenset([
                (i.idx, i.changed_value) for i in self.changes
            ])
        ))

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


FP_MARGIN_OF_ERROR = 1.0 + 10 ** -8

if __name__ == '__main__':
    fis, items = construct_sample()

    old_weight = -1
    obj = ChangeableDataset(fis, changes=items)
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
            assert changeset.weight() <= FP_MARGIN_OF_ERROR * old_weight
        old_weight = changeset.weight()
