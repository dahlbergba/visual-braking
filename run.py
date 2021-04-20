from agentEnv import AgentEnv
from mga import Microbial
from tools import save, read
from matplotlib import pyplot as plt
import numpy as np
import time


# TASK PARAMETERS ---------------------------------------------------------------
Dt = 0.01
#target_size = [45, 55, 65, 75]
#initial_distance = [120, 135, 150, 165, 180, 205, 210]
#initial_velocity = [10, 11, 12, 13, 14, 15]
target_size = [55, 65]
initial_distance = [150, 165, 180]
initial_velocity = [12, 13]
trial_length = 10 # (sec)
optical_variable = 0
trials = len(target_size) * len(initial_distance) * len(initial_velocity) 

# CTRNN PARAMETERS --------------------------------------------------------------
Size = 2    # Four interneurons plus one motor neuron
WeightRange = 16
BiasRange = 16
TimeConstMin = 1
TimeConstMax = 10
InputWeightRange = 16
GenotypeLength = Size*Size + Size*3


# EA PARAMETERS -----------------------------------------------------------------
GenotypeLength = Size*Size + Size*3    # Slightly longer because of incoding the input weight vector
Population = 5
RecombProb = 0.5
MutatProb = 0.1
Generations = 10
Tournaments = Generations * Population

# FITNESS FUNCTIONS -------------------------------------------------------------

def fitnessFunction1(genotype):
    """First version of the fitness function. Here we evolve agents simply on their ability to end 
    the trial near the target - crashes count, too!"""
    
    final_distances = np.empty(trials)
    i = 0
    
    agent = AgentEnv(genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt)
    for ts in target_size:
        for d in initial_distance:
            for v in initial_velocity:  
                agent.setInitialState(v, d, ts)
                agent.Optical_variable = optical_variable
                # While distance is still positive, agent is still moving forward significantly, and not too much time has elapsed
                while (agent.Distance > 0) and (agent.Velocity > 0.005) and (agent.Time < trial_length):
                    agent.sense()
                    agent.think()
                    agent.act()
    
                final_distances[i] = agent.Distance
                i += 1

    return np.average(final_distances)


# ============================= RUNTIME ========================================
    
start = time.time()
#np.random.seed(2)
print('Number of Evaluation Trials: %i' % trials)    

population = Microbial(fitnessFunction1, Population, GenotypeLength, RecombProb, MutatProb)
population.run(Tournaments)
filename = 'FinalDistance_G%i_V%i_%s' % ( int(population.generationsRun), optical_variable, population.dateCreated )
save(filename, population)

population.showFitnessSummary(('%s_Summary.png' % filename))
population.showFitnessTrajectories(('%s_Trajectories.png' % filename), _alpha=0.1)
       
print('TOTAL TIME ELAPSED: %i sec' % (int(time.time()-start)))
