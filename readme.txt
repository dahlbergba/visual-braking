README ----

The goal of this project is to evolve a visually guided braking agent, as proposed in
Kadhihasanoglu et al. 2015. After that, I'll be extending the work by applying several
perturbations to the agent: perturbing the agent's mapping to brake force, position,
and adding a delay between action and perception.



NEXT GOALS:
    >> Evolve agents to use each of the five kinds of visual information. 
        >>Try this with the acceleration Dt change.
    >> Evolve agents that have access to all five kinds of visual information. 
        >>Try this with the acceleration Dt change. 
    >> Conduct information analysis on optical variables and deceleration
    >> Evolve agents with perturbations and see how agents handle them. Are some strategies
    more resilient than others?
    

TO DO:



IN PROGRESS:



DONE:
    >> I am not sure which values to initialize each optical variable as. This is
    something I should look to Didem's code for guidance.
        >>> To accomplish this, I added a setInitialState method to the agent-env class that simulates
        a few moments of constant motion, ending at the proper/desired distance. 
    >> Currently not evolving. Population fitness is static on an individual level. Next step is to see if
    genes are changing. Also, it looks like no input is being passed to the agent. 
        >>> We're in business now. Within the agentEnv class, the agent's motor output was being stored
        in  self.acceleration rather than self.Acceleration. I fixed that and now it works. 
    >> Demonstrate evolvability of basic version of task - move forward as much as possible by end of trial. 
        >>> With a minimal population (P=10), this evolved in about 5 generations. I ran it to 25. See
        FinalDistance_G25_V0_2021-04-20. 
    >> Demonstrate evolvability of the next version of the task: final distance, crashes don't count. 
    >> Dt bug: the acceleration values WITH dt scaling vary from 0 to -0.3. Without, 0 to -3.0. But in KBB15, the 
    authors reported deceleration from 0 to -3. Therefore, I think the Dt scaling is not necessary in my code and was
    only needed in the KBB15 code because of the way their CTRNN class worked. 
    >> Agents evolved with only access to one optical variable at a time. 