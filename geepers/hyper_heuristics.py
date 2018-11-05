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
from geepers.data_manager import DataManager
from geepers.classifier import Classifier


class HyperHeuristic:
    def __init__(self, population_size, num_gens, crossover_prob, mutation_prob, chromosome_length, tournament_size, sample_size):
        self.population_size = population_size
        self.num_gens = num_gens
        self.crossover_probability = crossover_prob
        self.mutation_probability = mutation_prob
        self.chromosome_length = chromosome_length
        self.tournament_size = tournament_size
        self.sample_size = sample_size

        self.dm = DataManager()
        self.dm.load_dataset()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        definer = Classifier(data_manager=self.dm)
        definer.create_structure()
        self.classifier_pset = definer.get_structure()
        toolbox = base.Toolbox()
        toolbox.register("compile", gp.compile, pset=self.classifier_pset)
        self.toolbox = toolbox
        self.classifiers = []

    def load_classifiers(self):
        classifiers = []
        for i in range(20):
            with open('classifiers/{}.pkl'.format(i), 'rb') as file:
                data = pickle.load(file)
            classifiers.append(data)

        classifiers = sorted(classifiers, key=lambda x: x['hof'][0].fitness, reverse=True)[:5]
        for c in classifiers:
            c['func'] = self.toolbox.compile(expr=c['hof'][0])
        self.classifiers = classifiers
        return self

    def init_toolbox(self):
        creator.create("Combo", array.array, typecode='i', fitness=creator.FitnessMax)
        self.toolbox.register('hh_sample', random.randrange, 0, len(self.classifiers))
        self.toolbox.register('individual', tools.initRepeat, creator.Combo, self.toolbox.hh_sample, self.chromosome_length)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('evaluate', self.eval_classification)
        self.toolbox.register('mate', tools.cxTwoPoint)
        self.toolbox.register('mutate', tools.mutUniformInt, low=0, up=len(self.classifiers) - 1, indpb=0.5)
        self.toolbox.register('select', tools.selTournament, tournsize=self.tournament_size)

    def get_accuracy(self, individual, records):
        results = [bool(self.classifiers[individual[index % len(individual)]]['func'](*record[:-1]))
                   for index, record in enumerate(records.values)]
        result = len(records[records.attack_type == results])
        return result

    def eval_classification(self, individual):
        records = self.dm.get_training_set().sample(n=self.sample_size)
        return self.get_accuracy(individual, records),

    def run(self):
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        print('Crossover Rate: {}, Mutation Rate: {}'.format(self.crossover_probability, self.mutation_probability))
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=self.crossover_probability, mutpb=self.mutation_probability,
                                       ngen=self.num_gens, stats=stats, halloffame=hof, verbose=True)

        return pop, log, hof, self.crossover_probability, self.mutation_probability


if __name__ == "__main__":
    for i in range(0, 5):
        try:
            print()
            print("Starting run #{}...".format(i))
            start_time = datetime.datetime.now()
            print("Start time: {}".format(str(start_time)))
            hh = HyperHeuristic(250, 30, random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), 10, 10, 10000)
            hh.dm.split_dataset(0.6, 0.2)
            hh.load_classifiers()
            hh.init_toolbox()
            pop, logbook, hof, crossover_probability, mutation_probability = hh.run()
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
            training_set, validation_set, testing_set = hh.dm.get_training_set(), hh.dm.get_validation_set(), hh.dm.get_testing_set()
            print("Training Accuracy: {}".format(hh.get_accuracy(hof[0], training_set) / len(training_set) * 100))
            print("Validation Accuracy: {}".format(hh.get_accuracy(hof[0], validation_set) / len(validation_set) * 100))
            print("Testing Accuracy: {}".format(hh.get_accuracy(hof[0], testing_set) / len(testing_set) * 100))
            print()
        except Exception as ex:
            print()
            print("=" * 50)
            print('UNEXPECTED ERROR OCCURRED DURING RUN #{}'.format(i))
            print(str(ex))
            print("=" * 50)
            print()

