import operator
import math
import random
import time

import numpy

from ativ2_v2 import NeuralForest

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Taxas e Constantes
TOURNSIZE = 3
EXP_MIN, EXP_MAX = (1, 8)
MUT_MIN, MUT_MAX = (0, 4)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def setupCreate():
    TEST_SIZE = random.uniform(0.05, 0.4)
    MAX_ITER = random.randint(1000, 5000)
    LAYERS_AMT = random.randint(4, 10)
    LAYERS = tuple([random.randint(10, 50) for i in range(LAYERS_AMT)])
    return [TEST_SIZE, MAX_ITER, LAYERS]

toolbox = base.Toolbox()
toolbox.register("expr", setupCreate)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.expr, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalSetup(individual):
    individual = individual[0]
    # Pega a configuração gerada e roda a rede neural.
    TEST_SIZE = individual[0]
    MAX_ITER = individual[1]
    LAYERS = individual[2]
    network = NeuralForest(TEST_SIZE, MAX_ITER, LAYERS)
    preds, score, report, matrix = network.evaluate()
    return score,

toolbox.register("evaluate", evalSetup)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

def main():
    random.seed(time.time())
    genN = 2
    popN = 5

    pop = toolbox.population(n=popN)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.15, genN, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(hof)

    input()
    return pop, log, hof

if __name__ == "__main__":
    main()
