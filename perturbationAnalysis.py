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


def perturbMapping(filename):
    pass

def perturbMotorDelay(filename):
    pass

def perturbPosition(filename, perturbations, jweight, traj_params):
    """Perturbations should be a vector of position perturbations to be applied at every step. The 
    length should be trial_length/Dt. """
    
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
                    
                # Simulation loop
                i = 0
                while (agent.Distance > 0) and (agent.Velocity > 0.005) and (agent.Time < trial_length):
                    agent.sense()
                    agent.think()
                    agent.act()
                    agent.Distance += perturbations[i]
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
    for i in range(len(series)):    # This is necessary because of an inconsistent error I was getting
        time = np.arange(0, trial_length, agent.Dt)
        l = len(series[i])
        time = time[:l]
        plt.plot(time, series[i])            
        ov_labels = ['Image size', 'Image expansion rate', 'Tau', 'Tau-dot', 'Proportional Rate']
        labels = ['Distance (m)', 'Velocity (m/sec)', 'Acceleration (m/sec^2)]', ov_labels[agent.Optical_variable] ]
        plt.xlabel('Time (sec)')
        plt.ylabel(labels[i])
        plt.show()

    fitness= ( (1-np.average(final_distances)) + (1-np.average(final_velocities)) )/2 - jweight*np.average(jerks)
    print('Original Fitness: %f' % data.fitnessFunction(data.bestIndividual))
    print('Perturbed Fitness: %f' % fitness)
    

def jerk(accelerations):   # Sub-function used for calculating total jerk in a series of accelerations
    jerk = 0
    for i in range(1, len(accelerations)):
        jerk += (accelerations[i] - accelerations[i-1])**2
    return jerk 


# =====================================    RUNTIME    ==============================================


interval = 5
small_perb = 2
big_perb = 6

position_perturbations = np.zeros(int(50/Dt))
position_perturbations[int(interval/Dt)] = big_perb
position_perturbations[int(interval*2/Dt)] = -1 * big_perb
position_perturbations[int(interval*3/Dt)] = small_perb
position_perturbations[int(interval*4/Dt)] = -1 * small_perb

traj_params = (10, 120, 45)    # (v, d, ts)
#def perturbPosition(filename, perturbations, jweight, traj_params):
perturbPosition('DistanceVelocityJerk_V2_P150_T168_G_G200_2021-04-29', position_perturbations, 0, traj_params)














