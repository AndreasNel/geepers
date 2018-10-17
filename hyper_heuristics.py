#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import pickle
import datetime
import numpy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import itertools
import operator
import array
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# ====================== CAPTURE DATA BEG ==================================
# Read the different field names
fieldnames = pd.read_csv('field_names.csv', header=None, usecols=[0], squeeze=True).tolist()
arg_mapper = {("IN" + str(i)): key for i, key in enumerate(fieldnames)}
# Read the different attack types
attacktypes = pd.read_csv('attack_types.csv', header=None, usecols=[0], squeeze=True).tolist()[:-2]
# Read the data, with the appropriate headings
dataset = pd.read_csv('training_set.csv', header=None, names=fieldnames, true_values=['normal', 'unknown'], false_values=attacktypes)
fieldnames = fieldnames[:-1]
dataset.drop(columns=['difficulty_level'], inplace=True)

le = LabelEncoder()
dataset.protocol_type = le.fit_transform(dataset.protocol_type)
dataset.service = le.fit_transform(dataset.service)
dataset.flag = le.fit_transform(dataset.flag)
dataset[dataset.columns.difference(['attack_type'])] = dataset[dataset.columns.difference(['attack_type'])].astype(float)
# ====================== CAPTURE DATA END ==================================

# ====================== GP DEFINE BEG ==================================
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


# Define a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", list(itertools.repeat(float, len(fieldnames) - 1)), bool, "IN")
pset.renameArguments(**arg_mapper)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)
pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)
# ====================== GP DEFINE END ==================================

# ====================== GET CLASSIFIERS BEG ==================================
classifiers = []
for i in range(20):
    with open('classifiers/{}.pkl'.format(i), 'rb') as file:
        data = pickle.load(file)
    classifiers.append(data)

classifiers = sorted(classifiers, key=lambda x: x['hof'][0].fitness, reverse=True)[:5]
for c in classifiers:
    c['func'] = toolbox.compile(expr=c['hof'][0])
# ====================== GET CLASSIFIERS END ==================================

# ====================== GA DEFINE BEG ==================================
creator.create("Combo", array.array, typecode='i', fitness=creator.FitnessMax)
toolbox.register('hh_sample', random.randrange, 0, len(classifiers))
toolbox.register('individual', tools.initRepeat, creator.Combo, toolbox.hh_sample, 10)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


def eval_classification(individual):
    records = dataset.sample(frac=0.5)
    results = [bool(classifiers[individual[index % len(individual)]]['func'](*record[:-1])) for index, record in enumerate(records.values)]
    result = len(records[records.attack_type == results])
    return result,


toolbox.register('evaluate', eval_classification)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutUniformInt, low=0, up=len(classifiers) - 1, indpb=0.5)
toolbox.register('select', tools.selTournament, tournsize=10)
# ====================== GA DEFINE END ==================================


def main():
    pop = toolbox.population(n=250)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=30,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


if __name__ == "__main__":
    for i in range(0, 5):
        try:
            print()
            print("Starting run #{}...".format(i))
            start_time = datetime.datetime.now()
            print("Start time: {}".format(str(start_time)))
            pop, logbook, hof = main()
            execution_time = datetime.datetime.now() - start_time
            print("Finished, saving data...")
            print("End time: {}, Execution Time: {}".format(str(datetime.datetime.now()), str(execution_time)))
            saved_data = {
                "hof": hof,
                "logbook": logbook,
                "population": pop,
            }
            with open("hyper_heuristics/{}.pkl".format(i), "wb") as save_file:
                pickle.dump(saved_data, save_file)
            print()
        except Exception as ex:
            print()
            print("=" * 50)
            print('UNEXPECTED ERROR OCCURRED DURING RUN #{}'.format(i))
            print(str(ex))
            print("=" * 50)
            print()

