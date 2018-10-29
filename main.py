#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import random
import operator
import itertools
import pickle
import datetime
# import matplotlib.pyplot as plt
# import networkx as nx
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from data_manager import DataManager


dm = DataManager()
dm.load_dataset()

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", [float, int, int, int] + list(itertools.repeat(float, len(dm.get_field_names()) - 5)), bool, "IN")

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

arg_mapper = {("IN" + str(i)): key for i, key in enumerate(dm.get_field_names())}
pset.renameArguments(**arg_mapper)
# floating point operators
# Define a protected division function
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)


# logic operators
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2


pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

pset.addPrimitive(operator.lt, [int, int], bool)
pset.addPrimitive(operator.gt, [int, int], bool)
pset.addPrimitive(operator.eq, [int, int], bool)
pset.addPrimitive(if_then_else, [bool, int, int], int)

# terminals
pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
for i in numpy.unique(dm.get_data_set()[['protocol_type', 'service', 'flag']]):
    pset.addTerminal(i, int)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def get_accuracy(individual, records):
    func = toolbox.compile(expr=individual)
    results = [bool(func(*record[:-1])) for record in records.values]
    result = len(records[records.attack_type == results])
    return result


def classify(individual):
    return get_accuracy(individual, dm.get_training_set().sample(frac=0.2)),


toolbox.register("evaluate", classify)
toolbox.register("select", tools.selDoubleTournament, fitness_size=50, parsimony_size=1.4, fitness_first=True)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'), 10))
toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), 10))


def main():
    pop = toolbox.population(n=250)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    crossover_probability = random.uniform(0.0, 1.0)
    mutation_probability = random.uniform(0.0, 1.0)
    print('Crossover Rate: {}, Mutation Rate: {}'.format(crossover_probability, mutation_probability))

    # 0.6 and 0.2
    final_pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, mutpb=mutation_probability, ngen=15, stats=stats, halloffame=hof, verbose=True)

    return pop, stats, hof, final_pop, logbook, crossover_probability, mutation_probability


if __name__ == "__main__":
    for i in range(0, 10):
        try:
            print()
            print("Starting run #{}...".format(i))
            start_time = datetime.datetime.now()
            print("Start time: {}".format(str(start_time)))
            dm.split_dataset(0.6, 0.2)
            pop, stats, hof, final_pop, logbook, crossover_probability, mutation_probability = main()
            execution_time = datetime.datetime.now() - start_time
            print("Finished, saving data...")
            print("End time: {}, Execution Time: {}".format(str(datetime.datetime.now()), str(execution_time)))
            saved_data = {
                "hof": hof,
                "logbook": logbook,
                "population": final_pop,
                "crossover_probability": crossover_probability,
                "mutation_probability": mutation_probability,
            }
            with open("classifiers/{}.pkl".format(i), "wb") as save_file:
                pickle.dump(saved_data, save_file)
            training_set, validation_set, testing_set = dm.get_training_set(), dm.get_validation_set(), dm.get_testing_set()
            print("Training Accuracy: {}".format(get_accuracy(hof[0], training_set) / len(training_set) * 100))
            print("Validation Accuracy: {}".format(get_accuracy(hof[0], validation_set) / len(validation_set) * 100))
            print("Testing Accuracy: {}".format(get_accuracy(hof[0], testing_set) / len(testing_set) * 100))
            print()
        except Exception as ex:
            print()
            print("=" * 50)
            print('UNEXPECTED ERROR OCCURRED DURING RUN #{}'.format(i))
            print(str(ex))
            print("=" * 50)
            print()

    # nodes, edges, labels = gp.graph(hof[0])
    # g = nx.Graph()
    # g.add_nodes_from(nodes)
    # g.add_edges_from(edges)
    # pos = nx.drawing.nx_pydot.pydot_layout(g, prog="dot")
    # nx.draw_networkx_nodes(g, pos)
    # nx.draw_networkx_edges(g, pos)
    # nx.draw_networkx_labels(g, pos, labels)
    # plt.show()
