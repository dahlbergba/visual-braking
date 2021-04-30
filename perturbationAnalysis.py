from agentEnv import AgentEnv
from tools import save, read
from matplotlib import pyplot as plt
import numpy as np


def jerk(accelerations):   # Sub-function used for calculating total jerk in a series of accelerations
    jerk = 0
    for i in range(1, len(accelerations)):
        jerk += (accelerations[i] - accelerations[i-1])**2
    return jerk 


# ==============================    ANALYSIS DATA CLASS    =========================================


class analyzedData():
    
    def __init__(self, filename):
        self.filename = filename
        self.data = read(filename)
        self.Optical_variable = int(filename[len(self.data.fitnessFunction.__name__)+2])
        
        
    def save(self):
        save(('%s_Analyzed' % self.filename), self)
        
        
    def popAnalysis(self):
        fits = np.zeros(self.data.popsize)
        for i in range(self.data.popsize):
            fits[i] = self.data.fitnessFunction(self.data.pop[i])
        self.top25FitnessMean = np.sort(fits)[self.data.popsize-25:]
        self.bestFitness = np.sort(fits)[-1]


    def trialAnalysis(self, traj_params=(), show=True):
        """Suggested value for jweight are 0 or 1000 (KBB15). traj_params should be a 
        tuple of (initial velocity, initial distance, target size)."""
                
        # Step 1: Get genotype of best individual
        genotype = self.data.bestIndividual  
        
        # Step 2: Run simulation    
        final_distances = np.empty(ntrials) 
        final_velocities = np.empty(ntrials)
        jerks = np.empty(ntrials)
        trials = 0
        crashes = 0
        early_stops = 0
        timeouts = 0
        agent = AgentEnv(genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt)
        for ts in target_size:
            for d in initial_distance:
                for v in initial_velocity:  
                    agent.setInitialState(v, d, ts)
                    agent.Optical_variable = self.Optical_variable
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
                        i += 1
                        acceleration.append(agent.Acceleration)    # Record data
                        distance.append(agent.Distance)
                        velocity.append(agent.Velocity)
                        optical.append(agent.Optical_info[agent.Optical_variable])
    
                    if show == True and (v, d, ts) == traj_params: # If this is the trial we want to visualize, record data for plotting
                        series = [distance, velocity, acceleration, optical]
                        time = agent.Time
    
                    if agent.Distance < 0:   # If agent crashed, reset distance to starting position
                        agent.Distance = d
                        crashes += 1
                    
                    if agent.Distance > 15:  # If agent stopped prematurely 
                        early_stops += 1
                    
                    if agent.Time > trial_length:  # If agent never stops/times out
                        timeouts += 1
                    
                    if agent.Velocity < 0:   # If agent finishes moving backwards, reset velocity to starting velocity
                        agent.Velocity = v
                     
                    final_distances[trials] = agent.Distance/d   # Record performance data for fitness function
                    final_velocities[trials] = agent.Velocity/v
                    jerks[trials] = jerk(acceleration)
                    trials += 1
                       
        # Plot data 
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
                plt.title('%s-Guided Agent' % (ov_labels[agent.Optical_variable]))
                plt.show()
        
        self.successes = trials - crashes - early_stops - timeouts
        self.crashes = crashes
        self.earlyStops = early_stops
        self.timeouts = timeouts



def perturbationAnalysis(self, ptype, perturbations, traj_params=(), show=True):
    """Valid value for ptype are 'Position', 'Velocity', 'Mapping', and 'Delay'. Perturbations 
    should be a vector of perturbation values to be iterated through and applied at every step. The 
    length should be trial_length/Dt. The exception for this is in the case of Delay perturbations,
    in which case perturbations should be an integer representing the number of time steps to delay 
    motor output by. Suggested value for jweight are 0 or 1000 (KBB15). traj_params should be a 
    tuple of (initial velocity, initial distance, target size)."""
        
    # Step 1: Get genotype of best individual
    genotype = self.data.bestIndividual  
        
    # Step 2: Run simulation    
    final_distances = np.empty(ntrials)
    final_velocities = np.empty(ntrials)
    jerks = np.empty(ntrials)
    trials = 0
    crashes = 0
    early_stops = 0
    timeouts = 0
    agent = AgentEnv(genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt)
    for ts in target_size:
        for d in initial_distance:
            for v in initial_velocity:  
                agent.setInitialState(v, d, ts)
                agent.Optical_variable = self.Optical_variable
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

                if show == True and (v, d, ts) == traj_params:  # If this is the trial we want to visualize, record data for plotting
                    series = [distance, velocity, acceleration, optical]
                    time = agent.Time

                if agent.Distance < 0:   # If agent crashed, reset distance to starting position
                    agent.Distance = d
                    crashes += 1
                
                if agent.Distance > 15:  # If agent stopped prematurely 
                    early_stops += 1
                
                if agent.Time > trial_length:  # If agent never stops/times out
                    timeouts += 1
                
                if agent.Velocity < 0:   # If agent finishes moving backwards, reset velocity to starting velocity
                    agent.Velocity = v
                 
                 
                final_distances[trials] = agent.Distance/d   # Record performance data for fitness function
                final_velocities[trials] = agent.Velocity/v
                jerks[trials] = jerk(acceleration)
                trials += 1
                   
    # Plot data 
    fitness= ( (1-np.average(final_distances)) + (1-np.average(final_velocities)) )/2 - 1000*np.average(jerks)
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
        print('Original Fitness: %f' % self.data.fitnessFunction(self.data.bestIndividual))
        print('Perturbed Fitness: %f' % fitness)
        
    successes = trials - crashes - early_stops - timeouts  
    if ptype=='Delay':
        self.delayPerturbStats = fitness, successes, crashes, early_stops, timeouts
        self.delayPerturbation = perturbations
    elif ptype=='Position':
        self.positionPerturbStats = fitness, successes, crashes, early_stops, timeouts
        self.positionPerturbation = perturbations
    elif ptype=='Velocity':
        self.velocityPerturbStats = fitness, successes, crashes, early_stops, timeouts
        self.velocityPerturbation = perturbations    
    elif ptype=='Mapping':
        self.mappingPerturbStats = fitness, successes, crashes, early_stops, timeouts
        self.mappingPerturbation = perturbations    
    else:
        print('Ptype argument not recognized!') 


#============================================    PARAMETERS     ====================================


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
mapping_perturbations = np.full(int(interval/Dt), 1)
for i in [1.5, 1.5, 1.0, 0.5, 0.5]:
    m = np.full(int(interval/Dt), i)
    mapping_perturbations = np.concatenate((mapping_perturbations, m))
l = 500 - mapping_perturbations.shape[0]
mapping_perturbations = np.concatenate((mapping_perturbations, np.full(l, 1)))
print(mapping_perturbations)
print(mapping_perturbations.shape)

# Delay perturbation
delay = 3

# Which set of initial parameters to use for trajectory plots?
traj_params = (10, 120, 45)    # (v, d, ts)

#perturbationAnalysis('DistanceVelocity_V4_test', 'Delay', delay, 0, traj_params)