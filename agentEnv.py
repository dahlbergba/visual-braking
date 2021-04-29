from ctrnn import CTRNN
import math 
import numpy as np
import matplotlib.pyplot as plt

class AgentEnv():
    """This class implements a slightly streamlined version of the visually-guided braking task 
    proposed in KBB15. It includes both the agent and the environment because of their simplicity,
    whereas normally each of these would be their own class. Another important design choice is that
    this class does not record trajectory data. This is because trajectories are fully deterministic 
    given a set of starting states; therefore recording this data is unnecessary and cumbersome."""

    def __init__(self, genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt):

        #CONTROLLER ATTRIBUTES
        self.Dt = Dt    # Integration step of task, CTRNN (sec)
        self.NN = CTRNN(Size)   # Number of interneurons plus output neurons
        self.NN.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax,InputWeightRange)
        
        #AGENT ATTRIBUTES
        self.Brake_constant = 3.    # Scale factor for motor neuron output --> brake force
        self.Brake_effectiveness = 1.    # This is a scale factor that can be perturbed
        self.output = 0     # Activation of motor neuron
        self.Acceleration = 0.   # Brake force that agent is applying (m*m/sec)
        self.Velocity = 0.    # Agent's velocity in the direction of the target (m/sec)
        
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
        
        
    def setInitialState(self, velocity, distance, target_size): 
        """Simulates a few moments of constant motion to initialize the optical variables. This is 
        required because the image expansion rate, tau-dot, and proportional rate are all based on 
        the derivative of another optical variable. Do NOT confuse with the CTRNN controller's 
        method initializeState()."""
        steps = 3   # 3 steps is all that is required for the values of the optical information to stabilize
        self.Velocity = velocity
        self.Distance = distance + (steps*self.Dt*self.Velocity)    # Set back initial distance some amount to allow for initial constant motion
        self.Target_size = target_size
        self.Acceleration = 0.
        self.Time = 0.
        self.NN.initializeState(np.zeros(self.NN.Size))
        for step in range(steps):
            self.sense()    # Calculate optical variables
            self.Distance -= self.Velocity * self.Dt    # Move
                    
        
    def sense(self): 
        """Update optical variables and pass to NN controller."""
        #Calculate optical variables
        image_size = math.atan(self.Target_size/self.Distance)        
        image_expansion_rate = image_size - self.Optical_info[0]     # Difference between image_size and the last image_size (i=0)
        tau = image_size / image_expansion_rate
        tau_dot = tau - self.Optical_info[2]
        PR = tau/tau_dot
        self.Optical_info = [image_size, image_expansion_rate, tau, tau_dot, PR]
        
    
    def think(self):
        """Integrate CTRNN controller and read output."""
        self.NN.Input = np.full(self.NN.Size, self.Optical_info[self.Optical_variable])   # Vector of length Size and full of the selected optical variable
        self.NN.step(self.Dt)
        motor_neuron = 0  # Pick which neuron is the motor neuron (this is arbitrary, really) 
        self.output = self.NN.Output[motor_neuron]   
    
    
    def act(self):  
        """Read output from CTRNN controller, calculate action, and update the agent and environment. """
        # Calculate acceleration
        self.Acceleration = (-1) * self.output * self.Brake_constant * self.Brake_effectiveness      # Note the (-1) inversion
        self.Velocity += self.Acceleration * self.Dt         # Calculate velocity         
        self.Distance -= self.Velocity * self.Dt            # Calculate distance; note the -= operator
        self.Time += self.Dt


    def showTrajectory(self, optical_variable, target_size, distance, velocity, trial_length=100):
        """This method runs one simulation given some starting conditions and plots many different 
        variables (both physical and optical) against time. trial_length is the maximum trial length
        in seconds."""
        Acceleration_history = []
        Velocity_history = []
        Distance_history = []
        Optical_history = []
        #Generate data
        self.setInitialState(velocity, distance, target_size)
        self.Optical_variable = optical_variable
        self.Time = 0
        i = 0 
        # While distance is still positive, agent is still moving forward significantly, and not too much time has elapsed
        while (self.Distance > 0) and (self.Velocity > 0.005) and (self.Time < trial_length):
            self.sense()
            self.think()
            self.act()
            Acceleration_history.append(self.Acceleration)
            Velocity_history.append(self.Velocity)
            Distance_history.append(self.Distance)
            Optical_history.append(self.Optical_info)
            
        #Plot physics data    
        time = np.arange(0, self.Time, self.Dt)
        data = [Acceleration_history, Velocity_history, Distance_history]
        labels = ['Acceleration (m/sec^2)]', 'Velocity (m/sec)', 'Distance (m)']
        for i in range(len(data)):    # This is necessary because of an inconsistent error I was getting
            try: 
                plt.plot(time, data[i])
            except ValueError:    # If the time vector is the wrong size, shorten it slightly and try again
                time = np.arange(0, self.Time-self.Dt, self.Dt)
                plt.plot(time, data[i])
                
            plt.xlabel('Time (sec)')
            plt.ylabel(labels[i])
            plt.show()

        # Plot optical data
        ov_labels = ['Image size', 'Image expansion rate', 'Tau', 'Tau-dot', 'Proportional Rate']
        Optical_history = np.array(Optical_history)
        for i in range(len(self.Optical_info)):
            plt.plot(time, Optical_history[:,i])
            plt.xlabel('Time (sec)')
            plt.ylabel(ov_labels[i])
            plt.show()
        
        return Acceleration_history, Velocity_history, Distance_history, Optical_history
    