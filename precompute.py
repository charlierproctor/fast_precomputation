from contextlib import contextmanager
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
        return "item({})".format(self.idx.__repr__())

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


class DataChangeset(object):

    class ChangedDataItem(object):

        def __init__(self, item):
            self.item = item
            self.change_idx = 0

        def __repr__(self):
            return repr(self.item) + " = " + str(self.current_value)

        @property
        def current_value(self):
            """
            what is the current value of the associated item?
            """
            return self.item.possible_changes[self.change_idx]

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
            return self.change_idx < len(self.item.possible_changes) - 1

        def change(self):
            """
            go ahead and perform the change
            """
            self.change_idx += 1

    def __init__(self, changing_items):
        self.changing_items = changing_items
        self.changes = [self.ChangedDataItem(item) for item in changing_items]

    def __repr__(self):
        return repr(self.changes) + \
            " prob={:.2f}, importance={}, weight={:.2f}".format(
                self.joint_probability(),
                self.importance(),
                self.weight(),
            )

    def joint_probability(self):
        """
        joint probability of this changeset (with the current values) occurring
        assume data change events are independent
        """
        prob = 1
        for change in self.changes:
            prob *= change.item.probability(change.current_value)
        return prob

    def importance(self):
        """
        importance is given by the total number of dependent function instances
        of the associated changing items
        """
        dependent_fis = set()
        for change in self.changes:
            dependent_fis |= change.item.dependent_fi
        return len(dependent_fis)

    def weight(self):
        """
        weight of this changeset taking the current values
        """
        return (
            self.importance() * self.joint_probability()
        )

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

                    # assess the joint probability of these new values
                    probabilities.append(self.joint_probability())

            else:
                probabilities.append(0)

        assert len(probabilities) == len(self.changes)

        if probabilities:
            self.changes[probabilities.index(max(probabilities))].change()
        return probabilities

    @staticmethod
    def important_datasets(fis, skip=set()):
        """
        Yield datasets in decreasing order of importance,
        where *importance* is the number of dependent function instances
        skip: datasets to skip (not yield)
        """

        def important_datasets_generator(fis):
            if len(fis) == 0:
                yield frozenset()
            else:
                # recursively generate datasets (sets of data items to change)
                for dataset in important_datasets_generator(fis[1:]):

                    # change data this function instance depends on
                    for dependent_data_item in fis[0].data:
                        yield frozenset(set({dependent_data_item}) | dataset)

                    # exclude this function instance
                    yield dataset

        for dataset in important_datasets_generator(fis):
            if dataset not in skip:
                skip.add(dataset)
                yield DataChangeset(dataset)

    @classmethod
    def generate(cls, fis):
        """
        enumerate all possible permutations of changes on the data items
        """
        dataset_iterator = cls.important_datasets(fis)
        active_changesets = [next(dataset_iterator)]

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

            # if this is the last changeset, add another to the mix
            if best_idx == len(active_changesets) - 1:
                active_changesets.append(next(dataset_iterator))

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

    return fis


if __name__ == '__main__':
    fis = construct_sample()

    for changeset in DataChangeset.generate(fis):
        print(changeset)
