#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from rl_glue import *  # Required for RL-Glue
from matplotlib import pyplot as plt
RLGlue("mountaincar", "sarsa_lambda_agent")

import numpy as np

if __name__ == "__main__":
    num_episodes = 1000
    num_runs = 50

    steps = np.zeros([num_runs,num_episodes])

    for r in range(num_runs):
        print "run number : ", r
        RL_init()
        
        for e in range(num_episodes):
            
            # print '\tepisode {}'.format(e+1)
            
            RL_episode(0)    
            steps[r,e] = RL_num_steps()
            
            
    np.save('steps',steps)
    
    ave_steps=np.zeros(num_episodes )
    
    #for j in range(num_episodes):
     #   for i in range(num_runs):
      #      ave_steps[j]+=steps[i][j]
       # ave_steps[j]=ave_steps[j]/num_runs
    
   # plt.plot(ave_steps)
    #plt.xlabel('Episode')
    #plt.ylabel('Steps per episode \naveraged over 50 runs')   
    #plt.show()    