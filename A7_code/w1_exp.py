#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Andrew Jacobsen, Victor Silva, Mohammad M. Ajallooeian
  Last Modified on: 16/9/2017

  Experiment runs 2000 runs, each 1000 steps, of an n-armed bandit problem
"""
import matplotlib.pyplot as plt
from rl_glue import *  # Required for RL-Glue
RLGlue("w1_env", "w1_agent")

import numpy as np
import sys

def getOptimalAction():
    return int( RL_env_message("get optimal action") )


def save_results(data, data_size, filename): # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))
def op_value_RL_start(init_value_array):#returns NumPy array,init_value:Numpy array
    return RL_agent_start(init_value_array)
    
    
    



if __name__ == "__main__":
    num_runs = 2000
    max_steps = 1000
    init_value=5
    IV_array=np.zeros(1)
    IV_array[0]=init_value

    # array to store the results of each step
    optimal_action_1 = np.zeros(max_steps)
    optimal_action_2 = np.zeros(max_steps)

    #print "\nPrinting one dot for every run: {0} total Runs to complete".format(num_runs)
    for k in range(num_runs):
        RL_init()

        RL_start()
        for i in range(max_steps):
            # RL_step returns (reward, state, action, is_terminal); we need only the
            # action in this problem
            info= RL_step()
            action= info[1]


            '''
            check if action taken was optimal
            you need to get the optimal action; see the news/notices
            announcement on eClass for how to implement this
            '''
            op=getOptimalAction()
            # update your optimal action statistic here
            if action==op:
                optimal_action_1[i]+=1
                
        start_action=op_value_RL_start(IV_array) #start_action:Numpy_array
        for i in range(max_steps):
            op_info=RL_env_step(start_action)
            start_action=RL_agent_step(op_info[0], op_info[1])
            op=getOptimalAction()
            if start_action[0]==op:
                optimal_action_2[i]+=1
            
            

        RL_cleanup()
        sys.stdout.flush()
    
    plt.plot(np.divide(optimal_action_1,2000.0))
    plt.plot(np.divide(optimal_action_2,2000.0))
    plt.xlabel('Steps')
    plt.ylabel('%Optimal action')    
    plt.show()
    
