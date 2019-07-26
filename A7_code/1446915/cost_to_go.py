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
from tiles3 import *
from mpl_toolkits.mplot3d import axes3d

import numpy as np

iht = IHT(4096)
if __name__ == "__main__":
    num_episodes = 1000

    
    RL_init()
        
    for e in range(num_episodes):
            
        print '\tepisode {}'.format(e+1)
            
        RL_episode(0) 
        
    RL_agent_message("cost to go")
    
    points=open('value','r')
    plot=np.zeros((50,50))
    
    x=np.zeros(50)
    y=np.zeros(50)
    i=0
    for point in points:
        plot[i]=np.array(point.strip().split())
        x[i]=(i * 1.7 / 50)-1.2
        y[i]=(i * 0.14 /50)-0.07
        i+=1
        
    fig = plt.figure()
    a = fig.add_subplot(111,projection='3d')
    a.plot_wireframe(x,y,plot)
    plt.show()
        