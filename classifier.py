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


class Classifier():
    def __init__(self, data_manager):
        self.pset = None
        self.dm = data_manager

    def create_structure(self):
        # Define a new primitive set for strongly typed GP
        pset = gp.PrimitiveSetTyped("MAIN", [float, int, int, int] + list(
            itertools.repeat(float, len(self.dm.get_field_names()) - 5)), bool, "IN")
        arg_mapper = {("IN" + str(i)): key for i, key in enumerate(self.dm.get_field_names())}
        pset.renameArguments(**arg_mapper)
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addPrimitive(operator.not_, [bool], bool)
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(protectedDiv, [float, float], float)
        pset.addPrimitive(operator.lt, [float, float], bool)
        pset.addPrimitive(operator.gt, [float, float], bool)
        pset.addPrimitive(operator.eq, [float, float], bool)
        pset.addPrimitive(if_then_else, [bool, float, float], float)
        pset.addPrimitive(operator.lt, [int, int], bool)
        pset.addPrimitive(operator.gt, [int, int], bool)
        pset.addPrimitive(operator.eq, [int, int], bool)
        pset.addPrimitive(if_then_else, [bool, int, int], int)
        pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
        for i in np.unique(self.dm.get_data_set()[['protocol_type', 'service', 'flag']]):
            pset.addTerminal(i, int)
        pset.addTerminal(False, bool)
        pset.addTerminal(True, bool)
        self.pset = pset
        return self

    def get_structure(self):
        return self.pset
