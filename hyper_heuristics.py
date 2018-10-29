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
import array
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from data_manager import DataManager
from classifier import Classifier

# ====================== CAPTURE DATA BEG ==================================
dm = DataManager()
dm.load_dataset()
# ====================== CAPTURE DATA END ==================================

# ====================== GP DEFINE BEG ==================================
definer = Classifier(data_manager=dm)
definer.create_structure()
pset = definer.get_structure()

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


def get_accuracy(individual, records):
    results = [bool(classifiers[individual[index % len(individual)]]['func'](*record[:-1])) for index, record in enumerate(records.values)]
    result = len(records[records.attack_type == results])
    return result


def eval_classification(individual):
    records = dm.get_training_set().sample(n=10000)
    return get_accuracy(individual, records),


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

    crossover_probability = random.uniform(0.0, 1.0)
    mutation_probability = random.uniform(0.0, 1.0)
    print('Crossover Rate: {}, Mutation Rate: {}'.format(crossover_probability, mutation_probability))
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, mutpb=mutation_probability, ngen=30,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof, crossover_probability, mutation_probability


if __name__ == "__main__":
    for i in range(0, 5):
        try:
            print()
            print("Starting run #{}...".format(i))
            start_time = datetime.datetime.now()
            print("Start time: {}".format(str(start_time)))
            dm.split_dataset(0.6, 0.2)
            pop, logbook, hof, crossover_probability, mutation_probability = main()
            execution_time = datetime.datetime.now() - start_time
            print("Finished, saving data...")
            print("End time: {}, Execution Time: {}".format(str(datetime.datetime.now()), str(execution_time)))
            saved_data = {
                "hof": hof,
                "logbook": logbook,
                "population": pop,
                "crossover_probability": crossover_probability,
                "mutation_probability": mutation_probability,
            }
            with open("hyper_heuristics/{}.pkl".format(i), "wb") as save_file:
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

