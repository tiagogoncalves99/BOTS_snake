from random import shuffle, choice, sample, random, randint
from operator import  attrgetter

from copy import deepcopy
import csv
import time
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K

from snake_selection import *
from snake_crossover import*
from snake_mutation import*

from Run_Game import *

class Individual:
    
    '''
    Our Individuals take 2 inputs: 
        units - architecture of the neural network (list with the number of neurons on each layer)
        weights - weights of the NN (if None - they are randomly generated)
    
    '''
    
    def __init__(self, 
        units, # architecture of the neural network
        weights, # weights of the NN
    ):
        if weights == None:
            self.weights = np.array([2*np.random.rand(units[i-1],units[i])-1 for i in range(1,len(units))], dtype=object) # randomly generate weights from -1 to 1
        else:
            self.weights = weights
        
        self.units=units
        
        self.model = create_model_from_units(self.units, self.weights) # create the Keras neural network for the individual
        
        self.fitness_score = self.evaluate() # to prevent the game from running twice
        
        self.fitness = self.fitness_score[0] # Saving the fitness value for the individual
        self.score = self.fitness_score[1] # Saving the score it achieved (number of apples it ate)

    def evaluate(self):
        
        return run_game_with_ML2(display, clock, self.model) # run the game with the individual's neural network
        

    def model(self):
        return self.model # return the individual's model if necessary
    
    def get_model_weights(self):
        
        '''
        When called, this function returns the model's weights without the Bias values, since we're not working with them
        This will be necessary for the evolution process, on the Population's class
        '''
        
        
        aux_weights = self.model.get_weights()
        return np.array([aux_weights[i*2] for i in range(len(self.units)-1)], dtype=object) # multiplicated by 2 because we don't need the bias arrays, so we just iterate over
                                                                            # the even positions, which have the weights

    def __len__(self):
        return len(self.units)

    def __getitem__(self, position):
        return self.units[position]

    def __repr__(self):
        return f"Individual(Architecture={self.units}); Fitness: {self.fitness}" # the weights matrixes are too big, so we stick with the architecture


class Population:
    
    '''
    Our Populations take 3 inputs: 
        size - number of individuals in the population
        optim - max or min
        indiv_units - architecture of the neural network for the pop's individuals
            
    '''
    
    
    def __init__(self, size, optim, indiv_units, **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        self.gen = 1
        self.timestamp = int(time.time())
        self.indiv_units = indiv_units
        
        for _ in range(size):
            
            indiv = Individual(units=indiv_units,weights = None) # initializing each of the individuals
            self.individuals.append(indiv) # appending them the individuals list of the Population
            
            
            
    def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism):
        
        '''
        Implemented on the classes - only with slight changes for our project.
        
        '''
        
        
        
        for gen in range(1,gens+1):
            new_pop = []
            
            print('\nGen', self.gen, 'evolving:')
            
            if elitism == True:
                if self.optim == "max":
                    elite = max(self.individuals, key=attrgetter("fitness"))
                elif self.optim == "min":
                    elite = min(self.individuals, key=attrgetter("fitness"))
 
            while len(new_pop) < self.size:
                
                different=False
                
                # checking if they are different parents
                while different==False:
                    
                    parent1, parent2 = select(self), select(self)
                    if (parent1.get_model_weights()[0]==parent2.get_model_weights()[0]).all()==False: # if there are differences on the first matrix then they're different (lighter check)
                        different=True
                
                
                # to assess if things are going well:
                print('Parent1:', parent1)
                print('Parent2:', parent2,'\n')
                
                # Crossover
                
                # Since our crossover functions are implemented to work with the weight arrays of matrixes, we call for each parent the
                # get_model_weights method - so we can extract the weights without the bias values
                
                
                
                if random.random() < co_p:
                    offspring1, offspring2 = crossover(parent1.get_model_weights(), parent2.get_model_weights()) 
                else:
                    offspring1, offspring2 = parent1.get_model_weights(), parent2.get_model_weights()
                    
                # Mutation
                if random.random() < mu_p:
                    offspring1 = mutate(offspring1)
                if random.random() < mu_p:
                    offspring2 = mutate(offspring2)
 
                new_pop.append(Individual(units = self.indiv_units, weights = offspring1))
                
                if len(new_pop) < self.size:
                    new_pop.append(Individual(units = self.indiv_units, weights = offspring2))
 
            if elitism == True:
                if self.optim == "max":
                    least = min(new_pop, key=attrgetter("fitness"))
                elif self.optim == "min":
                    least = max(new_pop, key=attrgetter("fitness"))
                    
                new_pop.pop(new_pop.index(least))
                new_pop.append(elite)

            self.log()
            self.individuals = new_pop

            if self.optim == "max":
                
                best_individual = max(self, key=attrgetter("fitness"))

                print(f'Best Individual: {best_individual}')
                
            elif self.optim == "min":
                
                best_individual = min(self, key=attrgetter("fitness"))
                
                print(f'Best Individual: {min(self, key=attrgetter("fitness"))}')            
            
            
            #Save the best model of the generation, so in the end we can play the game with the best snake:
            best_individual.model.save(f'best_model_pop{self.timestamp}.h5')
            
            self.gen += 1

    def probabilistic_evolve(self, gens, co_p, mu_p, elitism):
        
        '''
        Our own evolution process - based on the one developed in classes but with additional randomness
        
        At each selection, crossover and mutation operation, one of the 3 developed operators is randomly chosen.
        
        Selection: 1 - FPS, 2 - Tournament, 3 - Rank
        
        Crossover: 1 - Weights Swap, 2 - Arithmetic, 3 - Blend
        
        Mutation: 1 - Swap, 2 - Inversion, 3 - Box
        
        '''
        
        
        for gen in range(1,gens+1):
            new_pop = []
            
            print('\nGen', self.gen, 'evolving:')
            
            if elitism == True:
                if self.optim == "max":
                    elite = max(self.individuals, key=attrgetter("fitness"))
                elif self.optim == "min":
                    elite = min(self.individuals, key=attrgetter("fitness"))
 
            while len(new_pop) < self.size:
                
                different=False
                
                # checking if they are different parents
                while different==False:
                    
                    sel_type = randint(1,3) # randomly select the selection operator: 1 - FPS, 2 - Tournament, 3 - Rank
                    if sel_type==1:
                        parent1, parent2 = fps(self), fps(self)
                    if sel_type==2:
                        parent1, parent2 = tournament(self), tournament(self)
                    if sel_type==3:
                        parent1, parent2 = rank(self), rank(self)
                                        
                    if (parent1.get_model_weights()[0]==parent2.get_model_weights()[0]).all()==False: # if there are differences on the first matrix then they're different (lighter check)
                        
                        different=True
                        
                print('Parent1:', parent1)
                print('Parent2:', parent2,'\n')
                
                # Crossover
                if random.random() < co_p:
                    
                    co_type = randint(1,3) # randomly select the crossover operator: 1 - Weights Swap, 2 - Arithmetic, 3 - Blend
                    if co_type==1:
                        offspring1, offspring2 = weights_swap_co(parent1.get_model_weights(), parent2.get_model_weights())
                    if co_type==2:
                        offspring1, offspring2 = arithmetic_co(parent1.get_model_weights(), parent2.get_model_weights())
                    if co_type==3:
                        offspring1, offspring2 = blend_co(parent1.get_model_weights(), parent2.get_model_weights())
                            
                else:
                    offspring1, offspring2 = parent1.get_model_weights(), parent2.get_model_weights()
                    
                # Mutation
                
                
                if random.random() < mu_p:
                    
                    mu_type = randint(1,3) # randomly select the mutation operator: 1 - Swap, 2 - Inversion, 3 - Box
                    if mu_type==1:
                        offspring1 = swap_mutation(offspring1)
                    if mu_type==2:
                        offspring1 = inversion_mutation(offspring1)
                    if mu_type==3:
                        offspring1 = box_mutation(offspring1)
                
                
                if random.random() < mu_p:
                    
                    mu_type = randint(1,3) # randomly select the mutation operator: 1 - Swap, 2 - Inversion, 3 - Box
                    if mu_type==1:
                        offspring2 = swap_mutation(offspring2)
                    if mu_type==2:
                        offspring2 = inversion_mutation(offspring2)
                    if mu_type==3:
                        offspring2 = box_mutation(offspring2)
 
                new_pop.append(Individual(units = self.indiv_units, weights = offspring1))
                
                if len(new_pop) < self.size:
                    new_pop.append(Individual(units = self.indiv_units, weights = offspring2))
 
            if elitism == True:
                if self.optim == "max":
                    least = min(new_pop, key=attrgetter("fitness"))
                elif self.optim == "min":
                    least = max(new_pop, key=attrgetter("fitness"))
                    
                new_pop.pop(new_pop.index(least))
                new_pop.append(elite)

            self.log()
            self.individuals = new_pop

            if self.optim == "max":
                
                best_individual = max(self, key=attrgetter("fitness"))

                print(f'Best Individual: {best_individual}')
                
            elif self.optim == "min":
                
                best_individual = min(self, key=attrgetter("fitness"))
                
                print(f'Best Individual: {best_individual}')            
            
            
            #Save the best model of the generation:
            best_individual.model.save(f'best_model_pop{self.timestamp}.h5')
            
            self.gen += 1


    def log(self):
        
        '''
        To register the evolution process - a csv is saved with the following info for each individual:
            Generation | Architecture | Individual Fitness | Individual Game Score
            
        This will be useful for report analysis of results
        
        '''
        
        
        with open(f'run_{self.timestamp}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            for i in self:
                writer.writerow([self.gen, i.units, i.fitness, i.score])


    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

    def __repr__(self):
        return f"Population(size={len(self.individuals)})"


def create_model_from_units(units, weights):
    
    '''
    Function to create a Keras Model automatically with the architecture and the weights as inputs
    
    It automatically ignores the bias values
    
    units = [number_of_inputs, hidden_1, ..., hidden_n, number_of_outputs]
    
    weights = [W_1, W_2, ... , W_n], 
        where W1 is the weight matrix from input to hidden_1
        W2 is the weight matrix from hidden_1 to hidden_2, etc.
    
    '''
    
    
    if len(units) < 2:
        print("Error: Model needs 2 layers at least")
        return None
    
    model = Sequential() # our Neural Network
    added_weights = 0 # for later iterations
    layers = len(units)  # considering input layer and first hidden layer are created at the same time
    
    for i in range(1, layers):
        activation = 'relu'
        if i == layers-1:
            activation = 'softmax' # last layer - activation softmax
        if i == 1:
            model.add(Dense(units=units[i], activation=activation, input_dim=units[0]))
        else:
            model.add(Dense(units=units[i], activation=activation)) # add a dense layer with the number of neurons specified on 'units'
            
        weight = weights[i-1]
        added_weights += units[i-1]*units[i]
        model.layers[-1].set_weights((weight, np.zeros(units[i]))) # adding the weights to the keras model, and null biases.
        
    return model