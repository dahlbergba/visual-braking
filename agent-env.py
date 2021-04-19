from ctrnn import CTRNN
import math 
import numpy as np

class Agent():


    def __init__(self, genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt):

        #CONTROLLER ATTRIBUTES
        self.Dt = Dt    # Integration step of task, CTRNN (sec)
        self.NN = CTRNN(Size)   # Number of interneurons plus output neurons
        self.NN.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax,InputWeightRange)
        
        #AGENT ATTRIBUTES
        self.Brake_constant = 3.    # Scale factor for notor neuron output --> brake force
        self.Brake_effectiveness = 1.    # This is a scale factor that can be perturbed
        self.Acceleration = 0.   # Brake force that agent is applying (m*m/sec)
        self.Velocity = 0. 
        
        #ENVIRONMENT ATTRIBUTES
        self.Target_size = 0.    # Size of target (m)
        self.Distance = 0.   # Distance from agent to target (m) 

        # OPTICAL ATTRIBUTES
        self.Optical_variable = 0   # Which optical variable is this agent paying attention to? (0-4)
        self.Optical_info = [1.,        # (0) Image size
                             1.,        # (1) Image expansion rate
                             1.,        # (2) Tau
                             1.,        # (3) Tau-dot
                             1.]        # (4) Proportional rate (PR)
        
        # DATA
        self.Brakemap_history = []
        self.Acceleration_history = []
        self.Velocity_history = []
        self.Distance_history = []
        self.Optical_history = []
        
                
    def setInitialState(self, velocity, distance, target_size):  # Simulate a few moments of constant motion to initialize the optical variables
        steps = 3   # 3 steps is all that is required for the values of the optical information to stabilize
        self.Velocity = velocity
        self.Distance = distance + (steps*self.Dt*self.Velocity)    # Set back initial distance some amount to allow for initial constant motion
        self.Target_size = target_size
        for step in range(steps):
            self.sense()    # Calculate optical variables
            self.Distance -= self.Velocity * self.Dt    # Move
            
            print(step) 
            print(self.Distance)
            print(self.Optical_info)
            print('####################  \n')
        
        
    def sense(self): # Update optical variables and pass to NN controller
        
        #Calculate optical variables
        image_size = math.atan(self.Target_size/self.Distance)        
        image_expansion_rate = image_size - self.Optical_info[0]     # Difference between image_size and the last image_size (i=0)
        tau = image_size / image_expansion_rate
        tau_dot = tau - self.Optical_info[2]
        PR = tau/tau_dot
        self.Optical_info = [image_size, image_expansion_rate, tau, tau_dot, PR]
        
        # Feed relevant optical variable to NN
        self.NN.Input = np.full(self.NN.Size, self.Optical_info[self.Optical_variable])   # Vector of length Size full of the selected optical variable
    
    
    def think(self):    #   Integrate NN controller
        self.nn.step(self.Dt)
   
    
    def act(self):  # Fetch output from NN controller, calculate action, update agent and environment
        
        # Calculate acceleration
        motor = 0  # Which neuron is the motor neuron (this is arbitrary, really) 
        output = self.nn.Output[motor]
        self.acceleration = (-1) * self.Brake_constant * output * self.Brake_effectiveness
        
        # Calculate velocity 
        self.Velocity += self.Acceleration * self.Dt
        
        # Calculate distance
        self.Distance -= self.Velocity * self.Dt    # Note that the operatior is -= because distance is decreasing
        
        # Store data
        self.Brake_eff_history.append(self.Brake_effectiveness)
        self.Acceleration_history.append(self.Acceleration)
        self.Velocity_history.append(self.Velocity)
        self.Distance_history.append(self.Distance)
        self.Optical_history.append(self.Optical_info)
