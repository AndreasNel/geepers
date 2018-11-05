import numpy as np
import operator
import itertools
import random
from deap import gp


# Define a protected division function
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2


class Classifier:
    def __init__(self, data_manager):
        self.pset = None
        self.dm = data_manager
        """Dictionary used to rename arguments. Keys are the old names, values are the new names."""
        self.arg_mapper = {("IN" + str(i)): key for i, key in enumerate(self.dm.get_field_names())}
        """
        List of primitives to add to the functional set.
        Each value is a tuple of the form (<function>, [<parameter types>], <return type>).
        """
        self.primitives = [
            (operator.and_, [bool, bool], bool),
            (operator.or_, [bool, bool], bool),
            (operator.not_, [bool], bool),
            (operator.add, [float, float], float),
            (operator.sub, [float, float], float),
            (operator.mul, [float, float], float),
            (protectedDiv, [float, float], float),
            (operator.lt, [float, float], bool),
            (operator.gt, [float, float], bool),
            (operator.eq, [float, float], bool),
            (if_then_else, [bool, float, float], float),
            (operator.lt, [int, int], bool),
            (operator.gt, [int, int], bool),
            (operator.eq, [int, int], bool),
            (if_then_else, [bool, int, int], int),
        ]
        terminals = list([(i, int) for i in np.unique(self.dm.get_data_set()[['protocol_type', 'service', 'flag']])])
        terminals += [(False, bool), (True, bool)]
        """
        List of terminals to add to the terminal set.
        Each value is a tuple of the form (<value>, <data type>).
        """
        self.terminal_set = terminals
        """A list of data types representing the data types for each input value of a record."""
        self.input_types = [float, int, int, int] + list(itertools.repeat(float, len(self.dm.get_field_names()) - 5))
        """The final return type of the classifier."""
        self.return_type = bool

    def create_structure(self):
        """
        Creates the primitive set for this classifier by making use of strongly typed GP.
        :return: self
        """
        self.pset = gp.PrimitiveSetTyped("MAIN", self.input_types, self.return_type, "IN")
        self.rename_arguments()
        self.load_primitives()
        self.pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
        self.load_terminal_set()
        return self

    def rename_arguments(self):
        """
        Renames the arguments in the structure.
        :return: self
        """
        if self.arg_mapper:
            self.pset.renameArguments(**self.arg_mapper)
        return self

    def load_ephemerals(self):

        return self

    def load_terminal_set(self):
        """
        Adds the terminals of the classifier to the terminal set of the classifier's primitive set.
        :return: self
        """
        for t in self.terminal_set:
            self.pset.addTerminal(*t)
        return self

    def load_primitives(self):
        """
        Loads the primitives of this classifier into the functional set.
        :return:
        """
        for p in self.primitives:
            self.pset.addPrimitive(*p)
        return self

    def get_structure(self):
        """
        Gets the primitive set of the classifier.
        :return: The primitive set.
        """
        return self.pset
