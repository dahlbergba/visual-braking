from ctrnn import CTRNN


class Agent():


    def __init__(self, genotype, Size, WeightRange, BiasRange, TimeConstMin, TimeConstMax, InputWeightRange, Dt):

        self.Dt = Dt
        self.nn = CTRNN(Size)
        self.nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax,InputWeightRange)

    def sense(self, _input):    # Check for input for task, _input should be a vector of length(Size)
        self.nn.Input = _input
    
    def think(self):    # Step NN
        self.nn.step(self.Dt)
   
    def act(self):    # Check for action
        output_neuron = 0
        action_threshold = 0.5    # Up for tweaking!
        if self.nn.Output[output_neuron] > action_threshold: 
            return 1
        else: 
            return 0