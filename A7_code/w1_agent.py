#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
 
  agent does *no* learning, selects actions randomly from the set of legal actions
 
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

last_action = None # last_action: NumPy array

num_actions = 10

def agent_init():
    global last_action

    last_action = np.zeros(1) # generates a NumPy array with size 1 equal to zero

def agent_start(this_observation): # returns NumPy array, this_observation: NumPy array
    global last_action,action_times,estimate_values,op_values#,op_init
    #op_init=0
    action_times=np.zeros(10)
    estimate_values=np.zeros(10)
    op_values=np.zeros(10)
    if this_observation[0]!=0:
        for i in range(num_actions):
            op_values[i]= this_observation[0]
        local_action = np.zeros(1)
        local_action[0] = rand_in_range(num_actions)             
        return  local_action
        #last_action[0] = rand_in_range(num_actions)
    local_action = np.zeros(1)
    local_action[0] = rand_in_range(num_actions)    
    last_action[0]=local_action[0]
    return local_action


def agent_step(reward, this_observation): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global last_action
    #the action at this time step 
    taken_action=int(this_observation[0])
    #how many times this action has been taken in current run
    if np.sum(op_values)!=0:
        op_values[taken_action]=op_values[taken_action]+(reward-op_values[taken_action])/10.0
        # might do some learning here        
        last_action[0]=np.argmax(op_values)
        
        return last_action
    #the estimate the action value
    estimate_values[taken_action]=estimate_values[taken_action]+(reward-estimate_values[taken_action])/10.0
    # might do some learning here
    current_op_action=np.argmax(estimate_values)          
    epsilon=rand_in_range(10)
    if epsilon==0:
        last_action[0]=rand_in_range(num_actions )
    else:
        last_action[0]=current_op_action
    
    
            
    
    
    

    return last_action

def agent_end(reward): # reward: floating point
    # final learning update at end of episode
    taken_action=last_action[0]
    #how many times this action has been taken in current run
    action_times[taken_action]+=1
    #the estimate the action value
    estimate_values[taken_action]=estimate_values[taken_action]+(reward-estimate_values[taken_action])*0.1    
    
    
    return

def agent_cleanup():
    # clean up
    return

def agent_message(inMessage): # returns string, inMessage: string
    # might be useful to get information from the agent

    if inMessage == "what is your name?":
        return "my name is skeleton_agent!"
    elif inMessage == "Op value involve":
        
        return 
  
    # else
    return "I don't know how to respond to your message"

