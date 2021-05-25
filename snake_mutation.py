import numpy as np
from random import uniform, sample, choice, randint, random
from operator import attrgetter
from Run_Game import *

# def swap_mutation(offspring):
    
#     mutated = offspring.copy()
    
#     for weights in mutated: # going through the weight matrixes
        
#         # Get two mutation points
#         mut_points = sample(range(len(weights)), 2)
        
#         # Swap the weight arrays
#         swap1 = weights[mut_points[0]].copy()
#         swap2 = weights[mut_points[1]].copy()
        
#         weights[mut_points[0]]=swap2
#         weights[mut_points[1]]=swap1
        
#     return mutated       

def swap_mutation(offspring, max_swaps=10):
    
    """ Swap Mutation Implementation
    
    Randomly generates a number between 1 and max_swaps - number_swaps
    
    Then randomly selects the index of the weights that will be swapped:
        idx1_1 - index of the matrix (for point 1)
        idx2_1 - index of the array (for point 1)
        idx3_1 - index of the weight point (for point 1)
        
        idx1_2 - index of the matrix (for point 2)
        idx2_2 - index of the array (for point 2)
        idx3_2 - index of the weight point (for point 2)
    
    After randomly selecting the indexes of the 2 weights, they are swapped.
    This is performed number_swaps times.
    
    
    """
    
    mutated = offspring.copy()
    
    number_swaps = randint(1,max_swaps) # number of points to swap
    
    for i in range(number_swaps):
        
        # weight 1 for swap:
        idx1_1 = randint(1,len(offspring)) - 1
        idx2_1 = randint(1,len(offspring[idx1_1])) - 1
        idx3_1 = randint(1,len(offspring[idx1_1][idx2_1])) - 1        
        
        swap1 = offspring[idx1_1][idx2_1][idx3_1] # saving the value to swap
        
        # weight 2 for swap:
        idx1_2 = randint(1,len(offspring)) - 1
        idx2_2 = randint(1,len(offspring[idx1_2])) - 1
        idx3_2 = randint(1,len(offspring[idx1_2][idx2_2])) - 1      
        
        swap2 = offspring[idx1_2][idx2_2][idx3_2] # saving the value to swap
                
        mutated[idx1_1][idx2_1][idx3_1] = swap2 # swapping
        mutated[idx1_2][idx2_2][idx3_2] = swap1 # swapping
        
    return mutated


# def inversion_mutation(offspring):

#     mutated = offspring.copy()
    
#     for weights in mutated:
        
#         mut_points = sample(range(len(weights)), 2)
        
#         # This method assumes that the second point is after (on the right of) the first one
#         # Sort the list
#         mut_points.sort()
        
#         #Invert for the mutation
#         weights[mut_points[0]:mut_points[1]] = weights[mut_points[0]:mut_points[1]][::-1]

#     return mutated

def inversion_mutation(offspring):
    
    """ Inversion Mutation Implementation
    
    This function performs an inversion on a segment of a randomly selected array:
    
        idx1 - randomly selected matrix index
        idx2 - randomly selected array index
        
        mut_points - randomly selected point indexes
        The inversion will take place in the segment between the 2 points
        (the mut_points list is sorted for this purpose)
        
    This inversion is performed once.
    
    
    """
    
    
    
    mutated = offspring.copy()
      
    # randomly select the indexes of the array to perform the mutation
    idx1 = randint(1,len(offspring)) - 1
    idx2 = randint(1,len(offspring)) - 1
    
    #print('indexes',idx1, idx2)
    
    mut_points = sample(range(len(offspring[idx1][idx2])), 2) # selecting 2 indexes for the mutation points
    
    mut_points.sort() # sorting to guarantee [lower index, higher index]
    
    #print('mut_points:',mut_points)
    
    inverted = mutated[idx1][idx2][mut_points[0]:mut_points[1]+1][::-1]  # inverted sequence
    mutated[idx1][idx2][mut_points[0]:mut_points[1]+1] = inverted # updating the sequence on the offspring
    
    return mutated



# def box_mutation(offspring,mutation_step=0.1):

#     mutated = offspring.copy()

#     for weights in mutated:

#         for i in range(len(weights)):
#             rx = uniform(mutation_step*(-1), mutation_step)
#             weights[i] = weights[i] + rx

#     return mutated

def box_mutation(offspring, max_changes = 10, mutation_step=0.1):
 
    
    """ Box Mutation Implementation
    
    Randomly generates a number between 1 and max_changes - number_changes
    
    Then randomly selects the index of the weight that will be changed:
        idx1 - index of the matrix 
        idx2 - index of the array 
        idx3 - index of the weight point 

    Randomly generates a value for rx - from Uniform( -mutation_step, mutation_step)
    
        -> Adds rx to the weight value
    
    This is performed number_changes times.
    
    
    """
    
    mutated = offspring.copy()
    
    number_changes = randint(1,max_changes)
    
    for i in range(number_changes):
        
        # weight 1 for swap:
        idx1 = randint(1,len(offspring)) - 1
        idx2 = randint(1,len(offspring[idx1])) - 1
        idx3 = randint(1,len(offspring[idx1][idx2])) - 1        
        
        rx = uniform(mutation_step*(-1), mutation_step) # generating a value for rx
        
        #print('indexes:', idx1, idx2, idx3)
        #print(rx)
        
        new_value = mutated[idx1][idx2][idx3] + rx 
        mutated[idx1][idx2][idx3] = new_value # updating the weight value

        
    return mutated
