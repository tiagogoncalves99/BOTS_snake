from Snake_Game import *
#from Feed_Forward_Neural_Network import *

'''

Running the game with a keras Neural Network.
Mainly taken from: https://github.com/TheAILearner/Training-Snake-Game-With-Genetic-Algorithm/blob/master/Run_Game.py

The code was adapted to work with a Keras Model, and different fitness functions were introduced (run_game_with_ML2 and run_game_with_ML3)

Moreover, each run_game_with_ML function, besides returning the fitness value, also returns the highest score achieved by the snake.

'''


def run_game_with_ML(display, clock, my_model):
    
    max_score = 0
    avg_score = 0
    test_games = 1
    score1 = 0
    steps_per_game = 2000
    score2 = 0
    max_steps_per_food=200 # maximum number of steps without eating food

    for _ in range(test_games):
        snake_start, snake_position, apple_position, score = starting_positions()

        count_same_direction = 0
        prev_direction = 0

        for _ in range(steps_per_game):
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            predictions = []
            
            # Predicting the next direction with the Keras Neural Network:
            predicted_direction = np.argmax(my_model.predict_on_batch(np.array(
                [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0],
                  snake_direction_vector_normalized[0], apple_direction_vector_normalized[1],
                  snake_direction_vector_normalized[1]]).reshape(-1, 7))) - 1

            
            
            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction

            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
            if predicted_direction == -1:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction == 1:
                new_direction = np.array([-new_direction[1], new_direction[0]])

            button_direction = generate_button_direction(new_direction)

            next_step = snake_position[0] + current_direction_vector
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(next_step.tolist(),
                                                                                        snake_position) == 1:
                score1 += -150
                break

            else:
                score1 += 0

            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)

            if score > max_score:
                max_score = score

            if count_same_direction > 8 and predicted_direction != 0:
                score2 -= 1
            else:
                score2 += 2


    return score1 + score2 + max_score * 5000, max_score

def run_game_with_ML2(display, clock, my_model):
    
    max_score = 0
    avg_score = 0
    test_games = 1
    score1 = 0
    steps_per_game = 10000
    score2 = 0
    max_steps_per_food=300 # maximum number of steps without eating food
    score3 = 0

    for _ in range(test_games):
        snake_start, snake_position, apple_position, score = starting_positions()

        count_same_direction = 0
        prev_direction = 0
        nr_steps_no_food=0

        for _ in range(steps_per_game):
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            
            
            predicted_direction = np.argmax(my_model.predict_on_batch(np.array(
                [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0],
                  snake_direction_vector_normalized[0], apple_direction_vector_normalized[1],
                  snake_direction_vector_normalized[1]]).reshape(-1, 7))) - 1

            score_before = score
            
            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction

            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
            if predicted_direction == -1:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction == 1:
                new_direction = np.array([-new_direction[1], new_direction[0]])

            button_direction = generate_button_direction(new_direction)

            next_step = snake_position[0] + current_direction_vector
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(next_step.tolist(),
                                                                                        snake_position) == 1:
                score1 += -150
                break

            else:
                score1 += 0

            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)
            
            if score_before==score:
                nr_steps_no_food += 1
            
            if score_before<score:
                nr_steps_no_food = 0 # the snake ate an apple so we reset the counter
            
            if nr_steps_no_food==max_steps_per_food: # if the snake reaches max_steps_per_food without eating any apple we crash the game and penalize the snake
                score3 += 1
                nr_steps_no_food = 0
                break
                #print('200 steps without food')
                
            if score > max_score:
                max_score = score

            if count_same_direction > 8 and predicted_direction != 0:
                score2 -= 1
            else:
                score2 += 2
    
    
    #print(score1 + score2 + max_score * 5000 - score3 * 500)
    return score1 + score2 + max_score * 5000 - score3 * 1000, max_score

def run_game_with_ML3(display, clock, my_model):
    
    max_score = 0
    avg_score = 0
    test_games = 1
    score1 = 0
    steps_per_game = 10000
    score2 = 0
    max_steps_per_food=300 # maximum number of steps without eating food
    score3 = 0

    for _ in range(test_games):
        snake_start, snake_position, apple_position, score = starting_positions()

        count_same_direction = 0
        prev_direction = 0
        nr_steps_no_food=0

        for _ in range(steps_per_game):
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            
            
            predicted_direction = np.argmax(my_model.predict_on_batch(np.array(
                [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0],
                  snake_direction_vector_normalized[0], apple_direction_vector_normalized[1],
                  snake_direction_vector_normalized[1]]).reshape(-1, 7))) - 1

            score_before = score
            
            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction

            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
            if predicted_direction == -1:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction == 1:
                new_direction = np.array([-new_direction[1], new_direction[0]])

            button_direction = generate_button_direction(new_direction)

            next_step = snake_position[0] + current_direction_vector
            
            if collision_with_self(next_step.tolist(),snake_position) == 1 and score > 3:
                
                score1 += -5000
                break
            
            else:
                score1 += 0
            
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(next_step.tolist(),
                                                                                        snake_position) == 1:
                score1 += -150
                break
            
            else:
                score1 += 0
            

            

            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)
            
            if score_before==score:
                nr_steps_no_food += 1
            
            if score_before<score:
                nr_steps_no_food = 0 # the snake ate an apple so we reset the counter
            
            if nr_steps_no_food==max_steps_per_food: # if the snake reaches max_steps_per_food without eating any apple we crash the game and penalize the snake
                score3 += 1
                nr_steps_no_food = 0
                break
                #print('200 steps without food')
                
            if score > max_score:
                max_score = score

            if count_same_direction > 8 and predicted_direction != 0:
                score2 -= 1
            else:
                score2 += 2
    
    
    #print(score1 + score2 + max_score * 5000 - score3 * 500)
    return score1 + score2 + max_score * 5000 - score3 * 1000, max_score