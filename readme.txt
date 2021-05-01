===================================================    OVERVIEW    =============================================================

The goal of this project is to evolve a visually guided braking agent, as proposed in
Kadhihasanoglu et al. 2015. After that, I'll be extending the work by applying several
perturbations to the agent: perturbing the agent's mapping to brake force, position,
and adding a delay between action and perception.





=================================================    ARCHITECTURE    ============================================================

Evolutionary runs on a local machine are performed by executing run.py, which contains all parameters and fitness functions
needed. This file calls on mga.py to create an instance of the Microbial class, which is what performs the steps of the 
evolutionary algorithm and stores the results. The fitness function calls on agentEnv.py to create an instance of agentEnv, 
which simulates the physics of the environment and the agent. Creating this instance in turn calls on ctrnn.py to create an
instance of CTRNN that serves as the controller for agentEnv. Data can be saved and read using the short function in tools.py. A
saved data file can be opened and analyzed using analysis.py. 

The run time is fairly long, so the evolutionary runs can also be performed using IU Carbonate. To do this, execute conditions.sh
on a Carbonate remote desktop. This will iterate a system argument from 0-9 and call on visualbraking.script each time, passing along
the system argument. This file submits a job to IU Carbonate consisting of running run_carbonate.py with the aforementioned system
argument, which specifies which of the 10 possible configurations of fitness function (2) and optical variable (5) to use. This 
has the advantage of running every evolutionary run in parallel, though runtimes are still very long (often >10 hrs). 



==============================================    PROGRESS TRACKING   ==========================================================


NEXT GOALS:
    How can the perturbation methods be more elegant and encompass more conditions?
    Why does the fitness function take so long?
        
    
TO DO:
    Debug fitness measures in analyzedData.trialAnalysis() method. 

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
    >> Fixed bug where fitness history was not being recorded in the runTournaments() method. 
    >> Go through and make sure everything is well-commented. 
    >> Check on whether a flush is needed for saving checkpointed data on Carbonate. 
        >>> It is NOT, because my save() function includes my_file.close(), which automatically 
        flushes the buffer. Yay foresight!(?)
    >> Conduct information analysis on optical variables and deceleration
    >> Evolve agents to use each of the five kinds of visual information. 
        >>> This was done but I'm going through now and checking for bugs. Some surprising results so far. 
        >>> Turns out 100 gens is enough to get recognizable braking behavior but its not very good. I have 
        sent several runs to Carbonate to try to get more generations.
        >>> I've set up what should be a FINAL run. It will run for 24 hours and save my data along
        the way, every 30 minutes.    
        >> Evolve agents with perturbations and see how agents handle them. 
    >> Develop brake force perturbation (probably in FF). 
    >> Develop position perturbation (probably in FF). 
    >> Develop delay perturbation (probably in FF). 