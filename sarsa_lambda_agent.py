from utils import rand_in_range, rand_un
import numpy as np
import pickle
from tiles3 import *
import random


numTilings=8
size=9*9*8*3
alpha=0.1/numTilings
e=0.0
gamma=1
deg_bootstap=0.9
iht = IHT(size)
 #1944 4096

def select_greedy_action(state):

    if np.random.random()<e:
        action=random.randint(3)
    else:
        v = np.zeros(3)
        for a in range(3):
            F = get_featureL(state,a)
            for i in F:
                v[a] += w[i]
        action = np.argmax(v)

    return action


def get_featureL(s,a):  #compute the feature value and do update. this part need to be revised.

    
    num_state1=1.7
    num_state2=0.14
    f1=s[0]/num_state1*8
    f2=s[1]/num_state2*8
    f=[f1,f2,a]
    featureL = tiles(iht,8,f)
    
    print(featureL)
    return featureL


def agent_init():
    global w
    ###select random # between 0 and -0.001 for init w.
    
    w=np.random.uniform(-0.001,0,size)

    return




def agent_start(state):

    global oldstate,trace_rate,oldaction,v

    trace_rate=np.zeros(size)#init eligi trace rate

    oldstate=state
    oldaction=select_greedy_action(oldstate)
    
    return oldaction


def agent_step(reward, state): 

    global w,trace_rate,oldstate,oldaction,v

    retur=reward


    F=get_featureL(oldstate,oldaction)
    for i in F:
        trace_rate[i]=1 #replacing trace
        retur=retur-w[i]
    action=select_greedy_action(state)



    F=get_featureL(state,action)
    for i in F:
        retur+=gamma*w[i]
    w+=alpha*retur*trace_rate
    trace_rate=gamma*deg_bootstap*trace_rate



    ###############################################

    oldstate=state
    oldaction=action

    return oldaction




def agent_end(reward):
    global trace_rate,w,oldstate,oldaction

    retur=reward


    F=get_featureL(oldstate,oldaction)
    for i in F:
        trace_rate[i]=1 
        retur=retur-w[i]
    w+=alpha*retur*trace_rate


    return

def agent_cleanup():

    return

def agent_message(in_message):
    if (in_message == "call function"): #for part 3 of A7.

        fout=open('value', 'w')
        steps=50

        for i in range(steps):
            for j in range(steps):

                values = []

                for a in range(3):

                    get_indices = get_featureL([-1.2 + (i * 1.7 / steps),-0.07 + (j * 0.14 / steps)],a)
                    get_sum=0

                    for index in get_indices:

                        get_sum += w[index]
                    values.append(-get_sum)
                height = np.max(values)
                fout.write(repr(height) + ' ')

            fout.write('\n')
        fout.close()
        return
