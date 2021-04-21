from agentEnv import AgentEnv
from mga import Microbial
from tools import save, read
from matplotlib import pyplot as plt
import numpy as np
import time


# TASK PARAMETERS ---------------------------------------------------------------
Dt = 0.1
target_size = [45, 55, 65, 75]
initial_distance = [120, 135, 150, 165, 180, 205, 210]
initial_velocity = [10, 11, 12, 13, 14, 15]
#target_size = [55, 65]
#initial_distance = [150, 165, 180]
#initial_velocity = [12, 13]
trial_length = 50 # (sec)
optical_variable = 4
trials = len(target_size) * len(initial_distance) * len(initial_velocity) 

# CTRNN PARAMETERS --------------------------------------------------------------
Size = 5    # Four interneurons plus one motor neuron
WeightRange = 16
BiasRange = 16
TimeConstMin = 1
TimeConstMax = 10
InputWeightRange = 16
GenotypeLength = Size*Size + Size*3 


# EA PARAMETERS -----------------------------------------------------------------
GenotypeLength = Size*Size + Size*3    # Slightly longer because of incoding the input weight vector
Population = 150
RecombProb = 0.5
MutatProb = 0.1
Generations = 200
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

    return (1 - np.average(final_distances))


def fitnessFunction2(genotype):
    """Second version of the fitness function. Here we evolve agents on their ability to end the 
    trial near the target, WITHOUT crashing. Crashes result in a default fitness of the initial 
    distance."""
    
    final_distances = np.empty(trials)
    i = 0
    
    agent = AgentEnv(genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt)
    for ts in target_size:
        for d in initial_distance:
            for v in initial_velocity:  
                agent.setInitialState(v, d, ts)
                agent.Optical_variable = optical_variable
                # While agent hasn't crashed, agent is still moving forward significantly, and not too much time has elapsed
                while (agent.Distance > 0) and (agent.Velocity > 0.005) and (agent.Time < trial_length):
                    agent.sense()
                    agent.think()
                    agent.act()
    
                if agent.Distance < 0:   # if agent crashed, reset distance to starting position
                    agent.Distance = d
                    
                final_distances[i] = agent.Distance/d   # fitness is proportion of distance
                i += 1

    return (1 - np.average(final_distances))


def fitnessFunction3(genotype):
    """Third version of the fitness function. This is the one used in Kadihasanoglu et al. 2015. 
    Here we evolve agents based on minimizing their final distance, final velocity, and sum jerk."""
    
    final_distances = np.empty(trials)
    final_velocities = np.empty(trials)
    jerks = np.empty(trials)
    i = 0
    
    agent = AgentEnv(genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt)
    for ts in target_size:
        for d in initial_distance:
            for v in initial_velocity:  
                agent.setInitialState(v, d, ts)
                agent.Optical_variable = optical_variable
                accelerations = []
                # While agent hasn't crashed, agent is still moving forward significantly, and not too much time has elapsed
                while (agent.Distance > 0) and (agent.Velocity > 0.005) and (agent.Time < trial_length):
                    agent.sense()
                    agent.think()
                    agent.act()
                    accelerations.append(agent.Acceleration)
    
                if agent.Distance < 0:   # if agent crashed, reset distance to starting position
                    agent.Distance = d
                    
                if agent.Velocity < 0: 
                    agent.Velocity = v
                    
                final_distances[i] = agent.Distance/d
                final_velocities[i] = agent.Velocity/v
                jerks[i] = jerk(accelerations)
                i += 1
                
    jweight = 0.
    fitness= ( (1-np.average(final_distances)) + (1-np.average(final_velocities)) )/2 - jweight*np.average(jerks)
    return fitness


def jerk(accelerations):
    jerk = 0
    for i in range(1, len(accelerations)):
        jerk += (accelerations[i] - accelerations[i-1])**2
    return jerk  

#   From didemsmain.cpp
#           decc2 = - Agent.NervousSystem.NeuronOutput(CIRCUITSIZE)*StepSize*3.0;                           
#		   //if (fabs(decc2 - decc1) > 0.005) {
#			   jerk += (decc1 - decc2)*(decc1 - decc2);
#		   //}
#	       decc1 = decc2;


# ============================= RUNTIME ========================================


# Set up    
start = time.time()
#np.random.seed(2)
print('Number of Evaluation Trials: %i' % trials)    

# Run evolutionary simulation 
#population = Microbial(fitnessFunction3, Population, GenotypeLength, RecombProb, MutatProb)
#population.run(Tournaments)
#filename = 'FinalDistanceNoCrash3_G%i_V%i_%s' % ( int(population.generationsRun), optical_variable, population.dateCreated )
population = Microbial(fitnessFunction3, Population, GenotypeLength, RecombProb, MutatProb)
population.run(Tournaments)
filename = 'DistanceVelocityJerk_G%i_V%i_%s' % ( int(population.generationsRun), optical_variable, population.dateCreated )
save(filename, population)

# Show graphs and save 
population.showFitnessSummary(('%s_Summary.png' % filename))
population.showFitnessTrajectories(('%s_Trajectories.png' % filename), _alpha=0.1) 
 
# Show trajectories of best individual
af, bf, sd, bi, bg, pf = population.fitStats()
agent = AgentEnv(bg, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt)
agent.showTrajectory(optical_variable, target_size[0], initial_distance[0], initial_velocity[0])

print('TOTAL TIME ELAPSED: %i sec' % (int(time.time()-start)))
