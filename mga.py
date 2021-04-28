import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from tools import save
import time

class Microbial():
    """This class is a modified version of a microbial genetic algorithm class that I got from 
    E. Izquierdo, which is in turn an implementation of the algorithm proposed in Harvey 2009. My 
    version of the class records some additional metadata (such as date edited), an additional 
    visualization method for plotting individual fitness trajectories, different statistics reported
    in fitStats(), and significant changes to run(). Now there are two options to running 
    tournaments: it can be done with a preset number of tournaments in mind (runTournaments()) or it
    can run endlessly, saving along the way, until terminated (runEndless()). Both versions report
    progress updates and fitness snapshots along the way. They both rely on the same underlying 
    method, tournament(), which fully constitutes Harvey's MGA."""
    
    
    def __init__(self, fitnessFunction, popsize, genesize, recombProb, mutatProb):
        self.fitnessFunction = fitnessFunction
        self.popsize = popsize
        self.genesize = genesize
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.pop = np.random.rand(popsize,genesize)*2 - 1
        self.medHistory = []    # List of median fitnesses, recorded each generation
        self.bestHistory = []   # List of best fitnesses, recorded each generation
        self.popHistory = []    # List of the entire population's fitnesses, recorded each generation
        self.generationsRun = 0
        self.dateCreated = str(date.today())
        self.dateEdited = str(date.today())


    def showFitnessSummary(self, savename=''):
        plt.plot(self.bestHistory, label="Best")
        plt.plot(self.medHistory, label="Average")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and Median Fitness")
        plt.legend()
        if savename !='':
            plt.savefig(savename)
        plt.show()
            
            
    def showFitnessTrajectories(self, savename='', _alpha=0.1):
        plt.plot(self.popHistory, alpha=_alpha)
        plt.plot(self.bestHistory, label="Best")
        plt.plot(self.medHistory, label="Median")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and Median Fitness")
        plt.legend()
        if savename !='':
            plt.savefig(savename)
        plt.show()

                        
    def fitStats(self):
        fits = np.zeros((self.pop.shape[0])) # Create empty vector for population fitnesses
        for i in range(self.pop.shape[0]):
            fitness = self.fitnessFunction(self.pop[i])
            fits[i] = fitness    # Populate vector
            
        iqr = np.percentile(fits, 75) - np.percentile(fits, 25)    # Interquartile range
        bi = fits.argmax()    # Best individual
        bg = self.pop[bi]       # Best genome
        return np.median(fits), np.max(fits), iqr, bi, bg, fits


    def tournament(self):
        # Step 1: Pick 2 individuals
        a = random.randint(0,self.popsize-1)
        b = random.randint(0,self.popsize-1)
        while (a==b):   # Make sure they are two different individuals
            b = random.randint(0,self.popsize-1)

        # Step 2: Compare their fitness
        if (self.fitnessFunction(self.pop[a]) > self.fitnessFunction(self.pop[b])):
            winner = a
            loser = b
        else:
            winner = b
            loser = a

        # Step 3: Transfect loser with winner
        for g in range(self.genesize):
            if (random.random() < self.recombProb):
                self.pop[loser][g] = self.pop[winner][g]

        # Step 4: Mutate loser and ensure new genes stay within bounds
        for g in range(self.genesize):
            self.pop[loser][g] += random.gauss(0.0,self.mutatProb)
            if self.pop[loser][g] > 1.0:
                self.pop[loser][g] = 1.0
            if self.pop[loser][g] < -1.0:
                self.pop[loser][g] = -1.0


    def runTournaments(self, tournaments, report=True):
        """Runs a set number of tournaments. Status reports can be suppresed by setting the keyword 
        report to False."""
        start = time.time()
        report_progress = 0    # Used later for reporting progress updates to console
        for i in range(tournaments):         # Evolutionary loop
            if (i%self.popsize==0):             # Every generation, record statistics
                mf, bf, iqr, bi, bg, pf = self.fitStats()
                self.medHistory.append(mf)
                self.bestHistory.append(bf)
                self.popHistory.append(pf)
                
            self.tournament()   # Run one tournament
            if (report==True) and (i/tournaments*100 > report_progress):    # If it is time (and if enabled) print status updates to console and iterate
                print(' \n%d%% Complete' % (report_progress))
                print('Fitness Median=%f, Max=%f, IQR=%f' % tuple(self.fitStats()[0:3]))
                print('Time elapsed: %f sec / %f min / %f hours' % ( (time.time()-start), (time.time()-start)/60, (time.time()-start)/3600 )) 
                report_progress += 10
        
        # After running evolutionary loop, update metadata and give final status update
        self.generationsRun += (tournaments/self.popsize)
        self.dateEdited = str(date.today())
        if report==True:
            print(' \n100% Complete')
            print('Fitness Median=%f, Max=%f, IQR=%f' % tuple(self.fitStats()[0:3]))
            print('Time elapsed: %f sec / %f min / %f hours' % ( (time.time()-start), (time.time()-start)/60, (time.time()-start)/3600 )) 
        
        
    def runEndless(self, filename, interval=15):
        """Run tournaments endlessly until terminated, and save data every so often. The interval between 
        saves/progress reports can be modifed by the keyword interval (minutes)."""
        start = time.time()
        last_report = start
        t = 0   # Counter for triggering generational recording of statistics
        while True:  # Evolutionary loop
            if (t%self.popsize==0):                 # Record statistics every generation
                af, bf, sd, bi, bg, pf = self.fitStats()
                self.medHistory.append(af)
                self.bestHistory.append(bf)
                self.popHistory.append(pf)
                self.generationsRun += (t/self.popsize)
                self.dateEdited = str(date.today())
                
            self.tournament()   # Run one tournament
            if (time.time()-last_report) > (interval*60):     # If it's been more than N minutes since last report/save, save/report
                print("\nSaving...")
                save(filename, self)
                print('Generations Run: %i' % int(self.generationsRun))     # Print generations run so far
                print('Fitness Median=%f, Max=%f, IQR=%f' % tuple(self.fitStats()[0:3]))
                print('Time elapsed: %f sec / %f min / %f hours' % ( (time.time()-start), (time.time()-start)/60, (time.time()-start)/3600 )) 
                last_report = time.time()
                
            t += 1