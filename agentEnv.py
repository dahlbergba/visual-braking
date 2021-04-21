from ctrnn import CTRNN
import math 
import numpy as np
import matplotlib.pyplot as plt

class AgentEnv():


    def __init__(self, genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt):

        #CONTROLLER ATTRIBUTES
        self.Dt = Dt    # Integration step of task, CTRNN (sec)
        self.NN = CTRNN(Size)   # Number of interneurons plus output neurons
        self.NN.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax,InputWeightRange)
        
        #AGENT ATTRIBUTES
        self.Brake_constant = 3.    # Scale factor for motor neuron output --> brake force
        self.Brake_effectiveness = 1.    # This is a scale factor that can be perturbed
        self.Acceleration = 0.   # Brake force that agent is applying (m*m/sec)
        self.Velocity = 0. 
        
        #ENVIRONMENT ATTRIBUTES
        self.Target_size = 0.    # Size of target (m)
        self.Distance = 0.   # Distance from agent to target (m)
        self.Time = 0.

        # OPTICAL ATTRIBUTES
        self.Optical_variable = 5   # Which optical variable is this agent paying attention to? (0-4)
        self.Optical_info = [1.,        # (0) Image size
                             1.,        # (1) Image expansion rate
                             1.,        # (2) Tau
                             1.,        # (3) Tau-dot
                             1.]        # (4) Proportional rate (PR)
        
        
    def setInitialState(self, velocity, distance, target_size):  # Simulate a few moments of constant motion to initialize the optical variables
        steps = 3   # 3 steps is all that is required for the values of the optical information to stabilize
        self.Velocity = velocity
        self.Distance = distance + (steps*self.Dt*self.Velocity)    # Set back initial distance some amount to allow for initial constant motion
        self.Target_size = target_size
        self.Acceleration = 0.
        self.Time = 0.
        for step in range(steps):
            self.sense()    # Calculate optical variables
            self.Distance -= self.Velocity * self.Dt    # Move
                    
        
    def sense(self): # Update optical variables and pass to NN controller
        #Calculate optical variables
        image_size = math.atan(self.Target_size/self.Distance)        
        image_expansion_rate = image_size - self.Optical_info[0]     # Difference between image_size and the last image_size (i=0)
        tau = image_size / image_expansion_rate
        tau_dot = tau - self.Optical_info[2]
        PR = tau/tau_dot
        self.Optical_info = [image_size, image_expansion_rate, tau, tau_dot, PR]
        
    
    def think(self):    #   Integrate NN controller
        self.NN.Input = np.full(self.NN.Size, self.Optical_info[self.Optical_variable])   # Vector of length Size and full of the selected optical variable
        self.NN.step(self.Dt)
   
    
    def act(self):  # Fetch output from NN controller, calculate action, update agent and environment
        # Calculate acceleration
        motor = 0  # Which neuron is the motor neuron (this is arbitrary, really) 
        output = self.NN.Output[motor]
        self.Acceleration = (-1) * output * self.Brake_constant * self.Brake_effectiveness
#        self.Acceleration = (-1) * output * self.Dt * self.Brake_constant * self.Brake_effectiveness
        
        # Calculate velocity 
        self.Velocity += self.Acceleration * self.Dt
        
        # Calculate distance
        self.Distance -= self.Velocity * self.Dt    # Note that the operatior is -= because distance is decreasing
        self.Time += self.Dt


    def record(self):         # Store data, this can be used when analyzing one particular run
        self.Brakemap_history.append(self.Brake_effectiveness)


    def showTrajectory(self, optical_variable, target_size, distance, velocity, trial_length=100, brakemap=True):
        Acceleration_history = []
        Velocity_history = []
        Distance_history = []
        Optical_history = []
        Brakemap_history = []
        #Generate data
        self.setInitialState(velocity, distance, target_size)
        self.Optical_variable = optical_variable
        self.Time = 0
        i = 0 
        # While distance is still positive, agent is still moving forward significantly, and not too much time has elapsed
        while (self.Distance > 0) and (self.Velocity > 0.005) and (self.Time < trial_length):
            if brakemap == False:   # If not using a constant brakemapping, then modify brakemapping and iterate
                self.brake_effectiveness = brakemap[i]
                i += 1
            self.sense()
            self.think()
            self.act()
            Acceleration_history.append(self.Acceleration)
            Velocity_history.append(self.Velocity)
            Distance_history.append(self.Distance)
            Optical_history.append(self.Optical_info)
            Brakemap_history.append(self.Brake_effectiveness)
            
        #Plot data    
        time = np.arange(0, self.Time, self.Dt)
        data = [Brakemap_history, Acceleration_history, Velocity_history, Distance_history]
        labels = ['Brake Mapping (%)', 'Acceleration (m/sec^2)]', 'Velocity (m/sec)', 'Distance (m)']
        for i in range(len(data)):
            try: 
                plt.plot(time, data[i])
            except ValueError:
                time = np.arange(0, self.Time-self.Dt, self.Dt)
                plt.plot(time, data[i])
                
            plt.xlabel('Time (sec)')
            plt.ylabel(labels[i])
            plt.show()

        ov_labels = ['Image size', 'Image expansion rate', 'Tau', 'Tau-dot', 'Proportional Rate']
        Optical_history = np.array(Optical_history)
        for i in range(len(self.Optical_info)):
            
            plt.plot(time, Optical_history[:,i])
            plt.xlabel('Time (sec)')
            plt.ylabel(ov_labels[i])
            plt.show()
        
        return Brakemap_history, Acceleration_history, Velocity_history, Distance_history, Optical_history
        
        
        
        
        
        
        
        
        
        