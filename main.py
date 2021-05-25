from Run_Game import *

from snake_charles import *

test_pop = Population(size=60, optim = 'max', indiv_units = [7,9,15,3])

#test_pop.evolve(gens=100, select = tournament, crossover = weights_swap_co, mutate = swap_mutation, co_p = 0.8, mu_p = 0.15, elitism = True)



test_pop.probabilistic_evolve(gens = 100, co_p=0.8, mu_p=0.15, elitism = False)



# test_individual1 = Individual(units=[7, 10,  4], weights=None)
# print(test_individual1.fitness)

# test_individual2 = Individual(units=[7, 10,  4], weights=None)
# print(test_individual2.fitness)


# test_individual3 = Individual(units=[7, 10,  4], weights=None)
# test_individual4 = Individual(units=[7, 10,  4], weights=None)
# test_individual5 = Individual(units=[7, 10,  4], weights=None)
# test_individual6 = Individual(units=[7, 10,  4], weights=None)
# test_individual7 = Individual(units=[7, 10,  4], weights=None)



# print(test_individual3.fitness)
# print(test_individual4.fitness)
# print(test_individual5.fitness)
# print(test_individual6.fitness)
# print(test_individual7.fitness)
