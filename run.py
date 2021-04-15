from agent import Agent
from mga import Microbial
from tools import save, read
from matplotlib import pyplot as plt
import numpy as np
import time

np.random.seed(2)

#TASK PARAMETERS ---------------------------------------------------------------
Duration = 1000    # Duration of simulation, in steps
Dt = 0.01
#N_task_agents = 2 # Number of copies of agent for the time task
Stimuli = [0.1, 0.2]    # External input/stimuli for the time task

#CTRNN PARAMETERS --------------------------------------------------------------
Size = 1
WeightRange = 15
BiasRange = 15
TimeConstMin = Dt*10
TimeConstMax = 5.0
InputWeightRange = 15

#EA PARAMETERS -----------------------------------------------------------------
Trials = 5   # N of trials per stimulus
GenotypeLength = Size*Size + Size*3    # Slightly longer because of incoding the input weight vector
Population = 100
RecombProb = 0.5
MutatProb = 0.1
Generations = 200
Tournaments = Generations * Population

#FITNESS FUNCTIONS -------------------------------------------------------------

def fitnessFunction1(genotype):
    """First version of the fitness function. This one optimizes simply for most actions taken and is 
    meant to help debug."""
    
    agent = Agent(genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt)
    first_actions = np.empty((len(Stimuli), Trials))
    for s in range(len(Stimuli)):    # This loops runs each stimulus condition
            for t in range(Trials):    # This loops run all trials of task
                actions = np.empty(Duration)
                agent.sense(Stimuli[s])    # Initial stimulus
                for step in range(Duration):    # Runtime
                    agent.think()
                    actions[step] = agent.act()    # Record agent action
                    agent.sense(0)    # Under current experimental design, no stimulus available for rest of 
                    
                first_actions[s, t] = np.count_nonzero(actions)
    return np.average(first_actions)/Duration


def fitnessFunction2(genotype):
    """Second version of the fitness function. This one optimizes for the greatest delay to action."""
    
    agent = Agent(genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt)
    first_actions = np.zeros((len(Stimuli), Trials))    # Changed to zeros for the sake of fitness eval in the event no action is taken
    for s in range(len(Stimuli)):    # This loops runs each stimulus condition
            for t in range(Trials):    # This loops run all trials of task
                actions = np.empty(Duration)
                agent.sense(Stimuli[s])    # Initial stimulus
                for step in range(Duration):    # Runtime
                    agent.think()
                    actions[step] = agent.act()    # Record agent action
                    agent.sense(0)    # Under current experimental design, no stimulus available for rest of 
                    
                acted = np.where(actions==1)
                if len(acted[0]) > 0:    # The 0 index is because we need the array's first size dimension
                    first_actions[s, t] = acted[0][0]   # Record when the agent's first action was 
                else: 
                    first_actions[s, t] = 0   # Not acting at all is worst fitness
                    
    return np.average(first_actions)/Duration


# ============================= RUNTIME ========================================
    
for s in range(10):
    Size = s + 1 
    GenotypeLength = Size*Size + Size*3
    start = time.time()
    subject = Microbial(fitnessFunction2, Population, GenotypeLength, RecombProb, MutatProb)
    subject.run(Tournaments)
    subject.showFitness2('FigD%i' % Size)
    save(('delayAction_N%i_P100' % Size), subject)
    print('TOTAL TIME ELAPSED: %f sec' % (time.time()-start))
