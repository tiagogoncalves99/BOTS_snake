import numpy as np
from random import uniform, sample, choice, randint, random
from operator import attrgetter
from Run_Game import *

def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
        
    Compatible with negative fitness values - 'adjustment' part
        If there are negative fitness values, it picks the minimum one and adds its absolute value to every individual's fitness
        So every fitness value is "dragged to the right". This will result in a minimum fitness value of 0, having every values positive
        
        It stops being exactly fitness proportionate, but the logic is the same
    
    """

    if population.optim == "max":
        
        min_fitness = min(population, key=attrgetter("fitness")).fitness # getting the minimum fitness value of the population
        
        if min_fitness<0: # If the minimum fitness is negative - Adjustment Part
            
            # Sum total fitnesses with adjustment
            total_fitness = sum([i.fitness + abs(min_fitness) for i in population])
            
            # Get a 'position' on the wheel
            spin = uniform(0, total_fitness)
            position = 0
        
            for individual in population:
                position += individual.fitness + abs(min_fitness)
                if position > spin:
                    return individual
        
        else: # Else we perform a normal FPS
            
            # Sum total fitnesses
            total_fitness = sum([i.fitness for i in population])
            # Get a 'position' on the wheel
            spin = uniform(0, total_fitness)
            position = 0
            # Find individual in the position of the spin
            for individual in population:
                position += individual.fitness
                if position > spin:
                    return individual
            
            
    elif population.optim == "min": 
        raise NotImplementedError

    else:
        raise Exception("No optimiziation specified (min or max).")




def tournament(population, size=10):
    # Select individuals based on tournament size
    tournament = sample(population.individuals, size)
    # Check if the problem is max or min
    if population.optim == 'max':
        return max(tournament, key=attrgetter("fitness"))
    elif population.optim == 'min':
        return min(tournament, key=attrgetter("fitness"))
    else:
        raise Exception("No optimiziation specified (min or max).")

def rank(population):
    # Rank individuals based on optim approach
    if population.optim == 'max':
        population.individuals.sort(key=attrgetter('fitness'))
    elif population.optim == 'min':
        population.individuals.sort(key=attrgetter('fitness'), reverse=True)

    # Sum all ranks
    total = sum(range(population.size+1))
    # Get random position
    spin = uniform(0, total)
    position = 0
    # Iterate until spin is found
    for count, individual in enumerate(population):
        position += count + 1
        if position > spin:
            return individual

