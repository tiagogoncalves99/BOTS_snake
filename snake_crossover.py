import numpy as np
from random import uniform, sample, choice, randint, random
from operator import attrgetter
from Run_Game import *

def weights_swap_co(parent1, parent2, max_swaps=25):
    
    """ Weights Swap Crossover Implementation
    
    Randomly generates a number between 1 and max_swaps - number_co_points
    
    Then randomly selects the index of the weights that will be swapped:
        idx1 - index of the matrix
        idx2 - index of the array
        idx3 - index of the weight point
    
    After randomly selecting the indexes of the weight (same index for parent 1 and parent2),
    the weights are swapped. This is performed number_co_points times.
    
    
    """
    
    
    number_co_points = randint(1,max_swaps) # number of crossover points
    
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    
    for i in range(number_co_points): # performed number_co_points times
        
        # randomly get a weight index to perform the crossover
        idx1 = randint(1,len(parent1)) - 1 # matrix index
        idx2 = randint(1,len(parent1[idx1])) - 1 # array index
        idx3 = randint(1,len(parent1[idx1][idx2])) - 1 # weight index
             
        swap1 = parent1[idx1][idx2][idx3] 
        swap2 = parent2[idx1][idx2][idx3] 
        
        offspring1[idx1][idx2][idx3] = swap2 # swapping value 1 with value 2
        offspring2[idx1][idx2][idx3] = swap1 # swapping value 2 with value 1
        
    return offspring1, offspring2



# def single_point_co(parents, offspring_size):

#     offspring = parents.copy()

#     for k in range(offspring_size[0]): 

#         while True:
#             parent1_idx = random.randint(0, parents.shape[0] - 1)
#             parent2_idx = random.randint(0, parents.shape[0] - 1)
#             # produce offspring from two parents if they are different
#             if parent1_idx != parent2_idx:
#                 p1 = list(parents[parent1_idx])
#                 p2 = list(parents[parent1_idx])
#                 co_point = randint(1, len(p1)-2)
#                 offspring1 = np.array(p1[:co_point] + p2[co_point:])
#                 offspring2 = np.array(p2[:co_point] + p1[co_point:])  
#                 offspring[parent1_idx] = offspring1
#                 offspring[parent2_idx] = offspring2 
#                 break
#     return offspring

def arithmetic_co(parent1, parent2, max_points=25):

    """ Arithmetic Crossover Implementation
    
    Randomly generates a number between 1 and max_points - number_co_points
    
    Then randomly selects the index of the weights that will be swapped:
        idx1 - index of the matrix
        idx2 - index of the array
        idx3 - index of the weight point
    
    Randomly selects a value for alpha from Uniform(0,1) distribution
    After randomly selecting the indexes of the weight (same index for parent 1 and parent2),
    the weights values are re-calculated from the expression:
        w1 = w1 * alpha + (1-alpha) * w2
        w2 = w2 * alpha + (1-alpha) * w1
    
    This is performed number_co_points times.
    
    
    """    


    number_co_points = randint(1,max_points)
    
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    
    for i in range(number_co_points):
        
        # randomly get a weight index to perform the crossover
        idx1 = randint(1,len(parent1)) - 1
        idx2 = randint(1,len(parent1[idx1])) - 1
        idx3 = randint(1,len(parent1[idx1][idx2])) - 1
        
        alpha = uniform(0,1) # select a random alpha between 0 and 1
        
        #print(idx1,idx2,idx3)
        #print(alpha)
        
        point1 = parent1[idx1][idx2][idx3] * alpha + (1 - alpha) * parent2[idx1][idx2][idx3] # new value for the weight on offspring 1
        point2 = parent2[idx1][idx2][idx3] * alpha + (1 - alpha) * parent1[idx1][idx2][idx3] # new value for the weight on offspring 2
        
        offspring1[idx1][idx2][idx3] = point1 # updating
        offspring2[idx1][idx2][idx3] = point2 # updating
        
    return offspring1, offspring2




# def blend_co(parent1,parent2,alpha=0.01):

#     # Offspring placeholders - None values make it easy to debug for errors
#     offspring1 = [None] * len(parent1)
#     offspring2 = [None] * len(parent2)

#     for i, (x1, x2) in enumerate(zip(parent1, parent2)):
#         gamma = (1. + 2. * alpha) * random.random() - alpha
#         offspring1[i] = (1. - gamma) * x1 + gamma * x2
#         offspring2[i] = gamma * x1 + (1. - gamma) * x2

#     return offspring1, offspring2

def blend_co(parent1,parent2,max_points=25,alpha=0.01):
    
    """ Blend Crossover Implementation
    
    Randomly generates a number between 1 and max_points - number_co_points
    
    Then randomly selects the index of the weights that will be swapped:
        idx1 - index of the matrix
        idx2 - index of the array
        idx3 - index of the weight point
    
    Randomly generates a value for gamma from the expression
        (1 + 2*alpha) * random.random() - alpha    , where alpha is a function parameter (default 0.01)
    
    random() generates a value between 0.0 and 1.0
    
    After randomly selecting the indexes of the weight (same index for parent 1 and parent2),
    the weights values are re-calculated from the expression:
        w1 = (1-gamma) * w1 + gamma * w2
        w2 = (1-gamma) * w2 + gamma * w1
    
    This is performed number_co_points times.
    
    
    """ 
    
    
    number_co_points = randint(1,max_points)
    
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    
    for i in range(number_co_points):
        
        # randomly get a weight index to perform the crossover
        idx1 = randint(1,len(parent1)) - 1
        idx2 = randint(1,len(parent1[idx1])) - 1
        idx3 = randint(1,len(parent1[idx1][idx2])) - 1
        
        #print('indexes:', idx1, idx2, idx3)       
        
        gamma = (1. + 2. * alpha) * random.random() - alpha # generating a random gamma
        
        x1 = offspring1[idx1][idx2][idx3] # saving the value of point 1
        x2 = offspring2[idx1][idx2][idx3] # saving the value of point 2
        
        #print('x1:',x1)
        #print('x2:',x2)
        
        point1 = (1. - gamma) * x1 + gamma * x2 # new value for point 1
        point2 = gamma * x1 + (1. - gamma) * x2 # new value for point 2
        
        #print('point1:', point1)
        #print('point2:', point2)
        
        offspring1[idx1][idx2][idx3] = point1 # updating
        offspring2[idx1][idx2][idx3] = point2 # updating
        
        #print('\n')
        
    return offspring1, offspring2


    
    
    