import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import time

class Microbial():
    
    def __init__(self, fitnessFunction, popsize, genesize, recombProb, mutatProb):
        self.fitnessFunction = fitnessFunction
        self.popsize = popsize
        self.genesize = genesize
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.pop = np.random.rand(popsize,genesize)*2 - 1
        self.avgHistory = []
        self.bestHistory = []
        self.popHistory = []
        self.generationsRun = 0
        self.dateCreated = str(date.today())
        self.dateEdited = str(date.today())


    def showFitnessSummary(self, savename=''):
        plt.plot(self.bestHistory, label="Best")
        plt.plot(self.avgHistory, label="Average")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        plt.legend()
        if savename !='':
            plt.savefig(savename)
        plt.show()

            
            
    def showFitnessTrajectories(self, savename='', _alpha=0.1):
        plt.plot(self.popHistory, alpha=_alpha)
        plt.plot(self.bestHistory, label="Best")
        plt.plot(self.avgHistory, label="Average")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        plt.legend()
        if savename !='':
            plt.savefig(savename)
        plt.show()

            
            
    def fitStats(self):
#        bestfit = self.fitnessFunction(self.pop[0])
#        bestind = -1
#        avgfit = 0
#        for i in self.pop:
#            fit = self.fitnessFunction(i)
#            avgfit += fit
#            if (fit > bestfit):
#                bestfit = fit
#                bestind = i
#         return avgfit/self.popsize, bestfit, popfit, bestind
        popfit = np.zeros((self.pop.shape[0])) # Vector of population fitnesses
        for i in range(self.pop.shape[0]):
            fitness = self.fitnessFunction(self.pop[i])
            popfit[i] = fitness
        
        bi = popfit.argmax()
        bg = self.pop[bi]
        
        return popfit.mean(), popfit.max(), popfit.std(), bi, bg, popfit


    def run(self, tournaments, report=True):

        # Evolutionary loop
        start = time.time()
        report_progress = 0
        for i in range(tournaments):

            # Report statistics every generation
            if (i%self.popsize==0):
                #print(i/self.popsize)
                #af, bf, bi = self.fitStats()
                af, bf, sd, bi, bg, pf = self.fitStats()
                self.avgHistory.append(af)
                self.bestHistory.append(bf)
                self.popHistory.append(pf)

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
            for l in range(self.genesize):
                if (random.random() < self.recombProb):
                    self.pop[loser][l] = self.pop[winner][l]

            # Step 4: Mutate loser and Make sure new organism stays within bounds
            for l in range(self.genesize):
                self.pop[loser][l] += random.gauss(0.0,self.mutatProb)
                if self.pop[loser][l] > 1.0:
                    self.pop[loser][l] = 1.0
                if self.pop[loser][l] < -1.0:
                    self.pop[loser][l] = -1.0

            if report==True:    # Print status updates to consode
                if i/tournaments*100 > report_progress:
                    print(' \n%d%% Complete' % (report_progress))
                    print('Fitness Mean=%f, Max=%f, SD=%f' % tuple(self.fitStats()[0:3]))
                    print('Time elapsed: %f sec / %f min / %f hours' % ( (time.time()-start), (time.time()-start)/60, (time.time()-start)/3600 )) 
                    report_progress += 10
        
        # Data integrity stuff
        self.generationsRun += (tournaments/self.popsize)
        self.dateEdited = str(date.today())
        