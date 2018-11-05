import random
import operator
import pickle
import datetime
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from data_manager import DataManager
from classifier import Classifier

dm = DataManager()
dm.load_dataset()
definer = Classifier(data_manager=dm)
definer.create_structure()
pset = definer.get_structure()

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
