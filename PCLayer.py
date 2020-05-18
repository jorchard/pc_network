
import numpy as np
from copy import deepcopy
import pickle
import torch
import matplotlib.pyplot as plt


dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device("cuda:0") # Uncomment this to run on GPU
else:
    device = torch.device("cpu")


class PCLayer:
    '''
     This PCLayer type is agnostic about being an erro node or a value node.
     Its identity as one or the other will be implicit in the connections
     and decays.
    '''

    def __init__(self, n=0):
        self.n = n    # Number of nodes

        # Node activities
        self.x = []          # state of the node
        self.dxdt = []       # derivatives of nodes (wrt time)
        self.x_decay = 0.    # decay of node
        self.tau = 0.1       # time constant
        self.batchsize = 0
        self.idx = -99       # index (for use in PCNetwork class)

        self.clamped = False

        # Probe variables
        self.probe_on = False
        self.x_history = []


    #=======
    # Setting behaviours
    def SetTau(self, tau):
        self.tau = tau

    def Clamped(self, is_clamped):
        '''
         lyr.Clamped(is_clamped)
         Clamps (True) or unclamps (False) the value stored in the layer.
         If it is clamped, the values are not updated.
        '''
        self.clamped = is_clamped

    #=======
    # Dynamics
    def RateOfChange(self, current):
        self.dxdt += current

    def Decay(self):
        '''
         lyr.Decay()
         Adds the decay term to the right-hand side of the differential
         equation, updating dxdt.
        '''
        self.dxdt -= self.x_decay*self.x

    def Step(self, dt=0.001):
        if not self.clamped:
            self.x += self.dxdt*dt/self.tau
        #self.dxdt.zero_()
        if self.probe_on:
            self.x_history.append(deepcopy(self.x.numpy()))


    #=======
    # Allocate and initialization
    def Allocate(self, batchsize=1):
        if batchsize!=self.batchsize:
            self.batchsize = batchsize
            del self.x, self.dxdt, self.x_history
            self.x_history = []
            self.x = torch.zeros(batchsize, self.n, dtype=torch.float32, device=device)
            self.dxdt = torch.zeros(batchsize, self.n, dtype=torch.float32, device=device)

    def Reset(self, random=0.):
        del self.x_history
        self.x_history = []
        if random==0.:
            self.x.zero_()
        else:
            self.x = torch.randn(self.x.shape[0], self.x.shape[1], dtype=torch.float32, device=device) * random
        self.dxdt.zero_()

    def SetDecay(self, x_decay):
        self.x_decay = x_decay

    def SetState(self, x):
        #self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.x = x.clone().detach()

    def Probe(self, bool):
        self.probe_on = bool
        if not self.probe_on:
            del self.x_history
            self.x_history = []


    #=======
    # Utilities
    def Plot(self, t_history, idx=0):
        if np.isscalar(idx):
            idx = [idx]
        if self.probe_on:
            for i in idx:
                plt.plot(t_history, np.array(self.x_history)[:,i,:])





# end
