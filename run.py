from agentEnv import AgentEnv
from mga import Microbial
from tools import save, read
from matplotlib import pyplot as plt
import numpy as np
import time


#==========================================    FITNESS FUNCTIONS    ============================================================


def FinalDistance(genotype):
    """First version of the fitness function. Here we evolve agents simply on their ability to end 
    the trial near the target - crashes count, too!"""
    
    final_distances = np.empty(ntrials)
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


def FinalDistanceNoCrash(genotype):
    """Second version of the fitness function. Here we evolve agents on their ability to end the 
    trial near the target, WITHOUT crashing. Crashes result in a default fitness of the initial 
    distance."""
    
    final_distances = np.empty(ntrials)
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


def DistanceVelocity(genotype):
    """Third version of the fitness function. This is the first one used in Kadihasanoglu et al. 2015. 
    Here we evolve agents based on minimizing their final distance and final velocity."""
    
    final_distances = np.empty(ntrials)
    final_velocities = np.empty(ntrials)
    jerks = np.empty(ntrials)
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
    
                if agent.Distance < 0:   # If agent crashed, reset distance to starting position
                    agent.Distance = d
                    
                if agent.Velocity < 0:   # If agent finishes moving backwards, reset velocity to starting velocity
                    agent.Velocity = v
                    
                final_distances[i] = agent.Distance/d
                final_velocities[i] = agent.Velocity/v
                jerks[i] = jerk(accelerations)
                i += 1
                
    jweight = 0.
    fitness= ( (1-np.average(final_distances)) + (1-np.average(final_velocities)) )/2 - jweight*np.average(jerks)
    return fitness


def DistanceVelocityJerk(genotype):
    """Fourth and final version of the fitness function. This is the second one used in Kadihasanoglu et al. 2015. 
    Here we evolve agents based on minimizing their final distance, final velocity, and average jerk."""
    
    final_distances = np.empty(ntrials)
    final_velocities = np.empty(ntrials)
    jerks = np.empty(ntrials)
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
    
                if agent.Distance < 0:   # If agent crashed, reset distance to starting position
                    agent.Distance = d
                    
                if agent.Velocity < 0:   # If agent finishes moving backwards, reset velocity to starting velocity
                    agent.Velocity = v
                    
                final_distances[i] = agent.Distance/d
                final_velocities[i] = agent.Velocity/v
                jerks[i] = jerk(accelerations)
                i += 1
                
    jweight = 1000.   # This value is the only difference from the above fitness function DistanceVelocity; empirical value from KBB15
    fitness= ( (1-np.average(final_distances)) + (1-np.average(final_velocities)) )/2 - jweight*np.average(jerks)
    return fitness


def jerk(accelerations):   # Sub-function used for calculating total jerk in a series of accelerations
    jerk = 0
    for i in range(1, len(accelerations)):
        jerk += (accelerations[i] - accelerations[i-1])**2
    return jerk 


#============================================    PARAMETERS     ================================================================


# TASK PARAMETERS
Dt = 0.1
#target_size = [45, 55, 65, 75]                            # Full set of parameters for evolutionary runs
#initial_distance = [120, 135, 150, 165, 180, 205, 210]    # Full set of parameters for evolutionary runs
#initial_velocity = [10, 11, 12, 13, 14, 15]               # Full set of parameters for evolutionary runs
target_size = [55, 65]               # Limited set of parameters for testing
initial_distance = [150, 165, 180]   # Limited set of parameters for testing
initial_velocity = [12, 13]          # Limited set of parameters for testing
trial_length = 50       # 50 is the KBB15 value (sec)
optical_variable = 0     # 0-4 are valid values, See AgentEnv class for glossary
fitnessFunction = DistanceVelocity
ntrials = len(target_size) * len(initial_distance) * len(initial_velocity) 


# CTRNN PARAMETERS
Size = 5    # Four interneurons plus one motor neuron
WeightRange = 16
BiasRange = 16
TimeConstMin = 1
TimeConstMax = 10
InputWeightRange = 16
GenotypeLength = Size*Size + Size*3 


# EA PARAMETERS
GenotypeLength = Size*Size + Size*3    # Slightly longer than usual because of encoding of the input weight vector
Population = 15    # KBB15 value is 150
RecombProb = 0.5
MutatProb = 0.1
Generations = 100 # No KBB15 value reported
Tournaments = Generations * Population


# ===========================================    RUNTIME    ====================================================================

for oi in range(0, 5):  # Run the below for each of the five optical variables
    # Set up  
    start = time.time()
    optical_variable = oi
    print('============  OPTICAL VARIABLE %i  ============' % oi)
    print('Number of Evaluation Trials: %i' % ntrials)    
    # Run simulation
    population = Microbial(fitnessFunction, Population, GenotypeLength, RecombProb, MutatProb)
    population.runTournaments(Tournaments)
    # Save data
    ffname = str(population.fitnessFunction.__name__)
    generation = int(population.generationsRun)
    popsize = population.popsize
    date = population.dateCreated
    filename = '%s_V%i_P%i_T%i_G%i_%s' % ( ffname, optical_variable, popsize, ntrials, generation, date)
    save(filename, population)
    #Report runtime
    print('TOTAL TIME ELAPSED: %i sec' % (int(time.time()-start))) 
#    # Show graphs and save them too
#    population.showFitnessSummary(('%s_Summary.png' % filename))
#    population.showFitnessTrajectories(('%s_Trajectories.png' % filename), _alpha=0.1)   
#    # Show trajectories of best individual
#    af, bf, sd, bi, bg, pf = population.fitStats()
#    agent = AgentEnv(bg, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt)
#    agent.showTrajectory(optical_variable, target_size[0], initial_distance[0], initial_velocity[0])
   
    
# RUN ENDLESSLY
#    population = Microbial(fitnessFunction, Population, GenotypeLength, RecombProb, MutatProb)
#     #Creat filename 
#    ffname = str(population.fitnessFunction.__name__)
#    generation = int(population.generationsRun)
#    popsize = population.popsize
#    date = population.dateCreated
#    filename = '%s_V%i_P%i_T%i_G%i_%s' % ( ffname, optical_variable, popsize, ntrials, generation, date)
#    # Run simulation and save
#    population.runEndless(filename, interval=0.5)    
#    #Report runtime
#    print('TOTAL TIME ELAPSED: %i sec' % (int(time.time()-start)))
