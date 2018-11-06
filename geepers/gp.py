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

from geepers.data_manager import DataManager
from geepers.classifier import Classifier


class GP:
    def __init__(self, population_size, num_gens, crossover_prob, mutation_prob, tournament_size, parsimony_size, max_height=None, sample_frac=1.0):
        self.population_size = population_size
        self.num_gens = num_gens
        self.crossover_probability = crossover_prob
        self.mutation_probability = mutation_prob
        self.tournament_size = tournament_size
        self.parsimony_size = parsimony_size
        self.max_height = max_height
        self.sample_frac = sample_frac

        self.dm = DataManager()
        self.dm.load_dataset()
        definer = Classifier(data_manager=self.dm)
        definer.create_structure()
        self.pset = definer.get_structure()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()

    def get_accuracy(self, individual, records):
        func = self.toolbox.compile(expr=individual)
        results = [bool(func(*record[:-1])) for record in records.values]
        result = len(records[records.attack_type == results])
        return result

    def classify(self, individual):
        return self.get_accuracy(individual, self.dm.get_training_set().sample(frac=self.sample_frac)),

    def init_toolbox(self):
        toolbox = self.toolbox
        pset = self.pset
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", self.classify)
        toolbox.register("select", tools.selDoubleTournament, fitness_size=self.tournament_size, parsimony_size=self.parsimony_size, fitness_first=True)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'), self.max_height))
        toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), self.max_height))
        return self

    def run(self):
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        print('Crossover Rate: {}, Mutation Rate: {}'.format(self.crossover_probability, self.mutation_probability))

        final_pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=self.crossover_probability,
                                                 mutpb=self.mutation_probability, ngen=self.num_gens, stats=stats, halloffame=hof,
                                                 verbose=True)

        return pop, stats, hof, final_pop, logbook, self.crossover_probability, self.mutation_probability


if __name__ == "__main__":
    for i in range(0, 10):
        try:
            print()
            print("Starting run #{}...".format(i))
            start_time = datetime.datetime.now()
            print("Start time: {}".format(str(start_time)))
            runner = GP(250, 15, random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), 50, 1.4, 10, 0.2)
            runner.dm.split_dataset(0.6, 0.2)
            runner.init_toolbox()
            pop, stats, hof, final_pop, logbook, crossover_probability, mutation_probability = runner.run()
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
            training_set, validation_set, testing_set = runner.dm.get_training_set(), runner.dm.get_validation_set(), runner.dm.get_testing_set()
            print("Training Accuracy: {}".format(runner.get_accuracy(hof[0], training_set) / len(training_set) * 100))
            print("Validation Accuracy: {}".format(runner.get_accuracy(hof[0], validation_set) / len(validation_set) * 100))
            print("Testing Accuracy: {}".format(runner.get_accuracy(hof[0], testing_set) / len(testing_set) * 100))
            print()
        except Exception as ex:
            print()
            print("=" * 50)
            print('UNEXPECTED ERROR OCCURRED DURING RUN #{}'.format(i))
            print(str(ex))
            print("=" * 50)
            print()
