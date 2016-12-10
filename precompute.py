from contextlib import contextmanager
import random
from scipy.stats import binom

MAX_VALUE = 10
NUM_FUNCTION_INSTANCES = 3
NUM_ITEMS = 3


class FunctionInstance(object):

    def __init__(self):
        self._data = set()

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


class ChangedDataItem(DataItem):

    def __init__(self, idx, value):
        super(ChangedDataItem, self).__init__(idx, value)

        self.change_idx = 0

    @classmethod
    def from_(cls, data_item):
        obj = cls(data_item.idx, data_item.value)
        obj.dependent_fi = data_item.dependent_fi
        return obj

    def __repr__(self):
        return "change(" + super(ChangedDataItem, self).__repr__() \
            + " ==> " + str(self.changed_value) + ")"

    @property
    def changed_value(self):
        """
        what is the current value of the associated item?
        """
        return self.possible_changes[self.change_idx]

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


class DataChangeset(object):

    def __init__(self, constants=None, changes=None):
        constants = frozenset() if constants is None else constants
        changes = frozenset() if changes is None else changes

        # sanity checks: sets are disjoint and contain data items
        assert constants.isdisjoint(changes)
        for item in constants | changes:
            assert isinstance(item, DataItem)

        self.constants = constants
        self.changes = [ChangedDataItem.from_(item) for item in changes]

    def __repr__(self):
        return "DataChangeset(constants={}, changes={})".format(
            repr(self.constants),
            repr(self.changes),
        )

    def joint_probability(self):
        return joint_probability(self.constants, self.changes)

    def importance(self):
        return importance(self.changes)

    def weight(self):
        return weight(self.constants, self.changes)

    def change(self):
        """
        change this dataset to its next best possible values
        return "false" when no changes left to make
        """
        probabilities = []
        for item in self.changes:
            if item.can_change():

                # fake the change (incrementing the change_idx)
                with item.fake_change():

                    # assess the joint probability of changing to new values
                    probabilities.append(
                        joint_probability(changes=self.changes)
                    )

            else:
                probabilities.append(0)

        assert len(probabilities) == len(self.changes)

        if sum(probabilities) > 0:
            self.changes[probabilities.index(max(probabilities))].change()
            return probabilities
        return False

    @staticmethod
    def important_datasets(fis, data):
        """
        Yield datasets in decreasing order of changeset_weight
        """

        def changeset_weight(changeset):
            return (
                # probability that other items remain constant
                joint_probability(constants=(data - changeset))

                # importance of changing these items
                * importance(changeset)
            )

        def important_changesets_generator(fis, depth=0):
            if len(fis) == 0:
                yield [frozenset()]
            else:
                # recursively generate changesets
                for changeset_group in important_changesets_generator(
                    fis[1:],
                    depth=depth + 1,
                ):

                    # goal: yield in decreasing order of changeset_weight:
                    # joint_probability(constants remaining current value)
                    # * importance(other items changing)

                    changesets = [
                        # change data this function instance depends on
                        frozenset(set({dependent_data_item}) | changeset)
                        for dependent_data_item in fis[0].data
                        for changeset in changeset_group
                    ]
                    # exclude this function instance
                    changesets += list(changeset_group)

                    changesets.sort(key=changeset_weight, reverse=True)

                    # yield the changesets in groups of the same weight
                    to_yield = []
                    for changeset in changesets:
                        if (
                            len(to_yield) > 0
                            and changeset_weight(changeset)
                            < changeset_weight(to_yield[-1])
                        ):
                            yield set(to_yield)
                            to_yield = []
                        to_yield.append(changeset)
                    yield set(to_yield)

        # skip changesets that have already been yielded
        skip = set()
        old_weight = 10000
        for changesets in important_changesets_generator(list(fis)):
            for y in changesets:
                assert changeset_weight(y) < old_weight
            old_weight = changeset_weight(list(changesets)[-1])

            changesets -= skip
            skip |= changesets

            yield frozenset([
                DataChangeset(data - changeset, changeset)
                for changeset in changesets
            ])

    @classmethod
    def generate(cls, fis, data):
        """
        enumerate all possible permutations of changes on the data items
        """
        dataset_iterator = cls.important_datasets(fis, data)
        active_changesets = list(next(dataset_iterator))

        while len(active_changesets) > 0:

            # find the changeset with the highest weight
            weights = [
                changeset.weight()
                for changeset in active_changesets
            ]
            best_idx = weights.index(max(weights))
            best_changeset = active_changesets[best_idx]

            # don't yield changesets without any probability
            if best_changeset.weight() > 0:
                yield best_changeset

            # if this is the last changeset, add more to the mix
            if best_idx == len(active_changesets) - 1:
                active_changesets += list(next(dataset_iterator))

            # make another change to this changeset for the future
            if not best_changeset.change():
                # we've exhausted all changes for this changeset
                active_changesets.remove(best_changeset)


def construct_sample():
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

    return frozenset(fis), frozenset(items)


if __name__ == '__main__':
    fis, data = construct_sample()

    old_weight = -1
    for changeset in DataChangeset.generate(fis, data):
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
