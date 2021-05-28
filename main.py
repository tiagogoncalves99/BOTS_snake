from Run_Game import *

from snake_charles import *

# test_individual1 = Individual(units=[7, 10,  4], weights=None)
# print(test_individual1.fitness)

# test_individual2 = Individual(units=[7, 10,  4], weights=None)
# print(test_individual2.score)


K.clear_session() #clear Keras session to free RAM space if necessary

test_pop1 = Population(size=60, optim = 'max', indiv_units = [7,9,15,3])
# #test_pop1.evolve(gens=100, select = tournament, crossover = weights_swap_co, mutate = swap_mutation, co_p = 0.8, mu_p = 0.15, elitism = True)
test_pop1.probabilistic_evolve(gens = 100, co_p=0.8, mu_p=0.15, elitism = True)

K.clear_session()


test_pop2 = Population(size=60, optim = 'max', indiv_units = [7,9,15,3])
test_pop2.evolve(gens=100, select = tournament, crossover = weights_swap_co, mutate = swap_mutation, co_p = 0.8, mu_p = 0.15, elitism = True)
#test_pop2.probabilistic_evolve(gens = 100, co_p=0.8, mu_p=0.15, elitism = False)
K.clear_session()






