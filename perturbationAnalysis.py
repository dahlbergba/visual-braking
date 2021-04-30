from agentEnv import AgentEnv
from mga import Microbial
from tools import save, read
from matplotlib import pyplot as plt
import numpy as np


#============================================    PARAMETERS     ================================================================


# TASK PARAMETERS
Dt = 0.1
target_size = [45, 55, 65, 75]                            # Full set of parameters for evolutionary runs
initial_distance = [120, 135, 150, 165, 180, 205, 210]    # Full set of parameters for evolutionary runs
initial_velocity = [10, 11, 12, 13, 14, 15]               # Full set of parameters for evolutionary runs
trial_length = 50       # 50 is the KBB15 value (sec)
ntrials = len(target_size) * len(initial_distance) * len(initial_velocity) 


# CTRNN PARAMETERS
Size = 5    # Four interneurons plus one motor neuron
WeightRange = 16
BiasRange = 16
TimeConstMin = 1
TimeConstMax = 10
InputWeightRange = 16
GenotypeLength = Size*Size + Size*3 



def perturbationAnalysis(filename, ptype, perturbations, jweight, traj_params, show=True):
    """Valid value for ptype are 'Position', 'Velocity', 'Mapping', and 'Delay'. Perturbations 
    should be a vector of perturbation values to be iterated through and applied at every step. The 
    length should be trial_length/Dt. The exception for this is in the case of Delay perturbations,
    in which case perturbations should be an integer representing the number of time steps to delay 
    motor output by. Suggested value for jweight are 0 or 1000 (KBB15). traj_params should be a 
    tuple of (initial velocity, initial distance, target size)."""
    
    # Step 1: Get optical variable data was evolved with (from filename)
    data = read(filename)
    ffnamelen = len(data.fitnessFunction.__name__)
    optical_variable = int(filename[ffnamelen+2])
    
    # Step 2: Get genotype of best individual
    genotype = data.bestIndividual  
    
    # Step 3: Run simulation    
    final_distances = np.empty(ntrials)
    final_velocities = np.empty(ntrials)
    jerks = np.empty(ntrials)
    trial = 0
    agent = AgentEnv(genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt)
    for ts in target_size:
        for d in initial_distance:
            for v in initial_velocity:  
                agent.setInitialState(v, d, ts)
                agent.Optical_variable = optical_variable
                acceleration = []           
                distance = []
                velocity = []
                optical = []
                if ptype == 'Delay':
                    outputs = []
                    for o in range(perturbations):
                        outputs.append(0)
                        
                # Simulation loop
                i = 0
                while (agent.Distance > 0) and (agent.Velocity > 0.005) and (agent.Time < trial_length):
                    
                    
                    agent.sense()
                    agent.think()
                    if ptype == 'Delay':
                        outputs.append(agent.output)   # Take self.output and append it to end of the list
                        agent.output = outputs.pop(0)  # Remove first element from list and set it to self.output
    
                    agent.act()
                    if ptype == 'Position':
                        agent.Distance += perturbations[i]
                    if ptype == 'Velocity':
                        agent.Velocity += perturbations[i]
                    if ptype == 'Mapping': 
                        agent.brake_effectiveness = perturbations[i]
                        
                    i += 1
                    acceleration.append(agent.Acceleration)    # Record data
                    distance.append(agent.Distance)
                    velocity.append(agent.Velocity)
                    optical.append(agent.Optical_info[agent.Optical_variable])


                if (v, d, ts) == traj_params:  # If this is the trial we want to visualize, record data for plotting
                    series = [distance, velocity, acceleration, optical]
                    time = agent.Time

                if agent.Distance < 0:   # If agent crashed, reset distance to starting position
                    agent.Distance = d
                    
                if agent.Velocity < 0:   # If agent finishes moving backwards, reset velocity to starting velocity
                    agent.Velocity = v
                 
                final_distances[trial] = agent.Distance/d   # Record performance data for fitness function
                final_velocities[trial] = agent.Velocity/v
                jerks[trial] = jerk(acceleration)
                trial += 1
                   
    # Plot data 
    fitness= ( (1-np.average(final_distances)) + (1-np.average(final_velocities)) )/2 - jweight*np.average(jerks)
    if show == True:
        for i in range(len(series)):    # This is necessary because of an inconsistent error I was getting
            time = np.arange(0, trial_length, agent.Dt)
            l = len(series[i])
            time = time[:l]
            plt.plot(time, series[i]) 
            ov_labels = ['Image size', 'Image expansion rate', 'Tau', 'Tau-dot', 'Proportional Rate']
            labels = ['Distance (m)', 'Velocity (m/sec)', 'Acceleration (m/sec^2)]', ov_labels[agent.Optical_variable] ]
            plt.xlabel('Time (sec)')
            plt.ylabel(labels[i])
            plt.title('%s Perturbation (%s agent)' % (ptype, ov_labels[agent.Optical_variable]))
            plt.show()
        print('Original Fitness: %f' % data.fitnessFunction(data.bestIndividual))
        print('Perturbed Fitness: %f' % fitness)
        
    return data.fitnessFunction(data.bestIndividual), fitness


def jerk(accelerations):   # Sub-function used for calculating total jerk in a series of accelerations
    jerk = 0
    for i in range(1, len(accelerations)):
        jerk += (accelerations[i] - accelerations[i-1])**2
    return jerk 


# =====================================    RUNTIME    ==============================================

# Creating position perturbations
interval = 2.5
small_perb = 2
big_perb = 5
position_perturbations = np.zeros(int(50/Dt))
position_perturbations[int(interval/Dt)] = big_perb
position_perturbations[int(interval*2/Dt)] = -1 * big_perb
position_perturbations[int(interval*3/Dt)] = small_perb
position_perturbations[int(interval*4/Dt)] = -1 * small_perb

# Creating velocity perturbations
interval = 2.5
small_perb = 2
big_perb = 5
velocity_perturbations = np.zeros(int(50/Dt))
velocity_perturbations[int(interval/Dt)] = big_perb
velocity_perturbations[int(interval*2/Dt)] = -1 * big_perb
velocity_perturbations[int(interval*3/Dt)] = small_perb
velocity_perturbations[int(interval*4/Dt)] = -1 * small_perb

# Creating mapping perturbations
mapping_perturbations = np.full((int(50/Dt)), 1)

# Delay perturbation
delay = 3

# Which set of parameters to use for trajectory plots?
traj_params = (10, 120, 45)    # (v, d, ts)

perturbationAnalysis('DistanceVelocity_V4_test', 'Delay', delay, 0, traj_params)