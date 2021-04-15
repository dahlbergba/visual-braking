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
        self.Optical_variable = 5   # Which optical variable is this agent paying attention to? (0-4)
        self.Optical_info = [math.atan(self.Target_size/self.Distance), # (0) Image size
                             0.,                                        # (1) Image expansion rate
                             999999.,                                   # (2) Tau
                             0.,                                        # (3) Tau-dot
                             999999.]                                   # (4) Proportional rate (PR)
        
        # DATA
        self.Brakemap_history = []
        self.Acceleration_history = []
        self.Velocity_history = []
        self.Distance_history = []
        self.Optical_history = []
        
                
    def sense(self):
        
        #Calculate optical variables
        image_size = math.atan(self.Target_size/self.Distance)        
        image_expansion_rate = image_size - self.Optical_info[0]     # Difference between image_size and the last image_size (i=0)
        tau = image_size / image_expansion_rate
        tau_dot = tau - self.Optical_info[2]
        PR = tau/tau_dot
        self.Optical_info = [image_size, image_expansion_rate, tau, tau_dot, PR]
        
        # Feed relevant optical variable to NN
        self.NN.Input = np.full(self.NN.Size, self.Optical_info[self.optical_variable])   # Vector of length Size full of the selected optical variable
    
    
    def think(self):    #   Integrate NN
        self.nn.step(self.Dt)
   
    
    def act(self):
        
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
