#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
 
  agent does *no* learning, selects actions randomly from the set of legal actions
 
"""


import numpy as np
import random
from tiles3 import * 

num_tilings=8
tilings_size=8*8
memory_size=4096
alpha=0.1/num_tilings
gamma=1.0
lamda=0.9
iht=IHT(memory_size)

def agent_init():
    global w_vector
    w_vector=np.zeros(memory_size)
    for i in range(len(w_vector)):
        w_vector[i]=random.uniform(-0.001,0)

def agent_start(this_observation): # returns 
    global w_vector, z_vector,last_action,last_state
    
    last_state=this_observation
    
    if np.random.random()<0:
        last_action=random.randint(3)    
    else:
        act_v=np.zeros(3)
        for i in range(3):
            x_vector=get_x_vector(i,this_observation)
            for j in x_vector:
                act_v[i]+=w_vector[j]
        last_action=np.argmax(act_v)        

   
    
    z_vector=np.zeros(memory_size)
    

    return last_action


def agent_step(reward, this_observation): # returns
    global w_vector, z_vector,last_action,last_state
    delta=reward
    x_vector=get_x_vector(last_action,last_state)
    
    for i in x_vector:
        delta=delta-w_vector[i]
        z_vector[i]=1
    if np.random.random()<0:
        new_action=random.randint(3)                
    else:
        act_v=np.zeros(3)
        for i in range(3):
            x_vector=get_x_vector(i,this_observation)
            for j in x_vector:
                act_v[i]+=w_vector[j]            
        new_action=np.argmax(act_v)        
           

    #print(action)    
    x_vector=get_x_vector(new_action,this_observation)
    for i in x_vector:
        delta=delta+gamma*w_vector[i]
    w_vector=w_vector+alpha*delta*z_vector
    z_vector=gamma*lamda*z_vector
    
    last_state=this_observation
    last_action=new_action
    
   
    return last_action

def agent_end(reward):
    global w_vector, z_vector,action
    delta=reward
    
    x_vector=get_x_vector(last_action,last_state)
    for i in x_vector:
        delta=delta-w_vector[i]
        z_vector[i]=1
    w_vector=w_vector+alpha*delta*z_vector
    
    
 
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
    elif inMessage=="cost to go":
        cost_to_go()
        
        return
       
        
  
    # else
    return "I don't know how to respond to your message"

def get_x_vector(action_value,state):#return tile with respect to input action value
    
    a=tiles(iht,num_tilings,[8*state[0]/(0.5+1.2),8*state[1]/(0.07+0.07),action_value])

    return a

def cost_to_go():
    fout=open('value', 'w')
    steps=50
    for i in range(steps):
        for j in range(steps):
            values = []
            for a in range(3):
                inds=get_x_vector(a,[-1.2 + (i * 1.7 / steps),-0.07 + (j * 0.14 / steps)])
                v_sum=0
                for k in inds:
                    v_sum+=w_vector[k]
                values.append(v_sum)
            height = np.max(values)
            fout.write(repr(-height) + ' ')
        fout.write('\n')
    fout.close()            
    
    return
            
    