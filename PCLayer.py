
import numpy as np
from copy import deepcopy
import pickle
import torch
import matplotlib.pyplot as plt


dtype = torch.float32
# if torch.cuda.is_available():
#     device = torch.device("cuda:5") # Uncomment this to run on GPU
# else:
#     device = torch.device("cpu")
global device

class PCLayer:
    '''
     This PCLayer type is agnostic about being an error node or a value node.
     Its identity as one or the other will be implicit in the connections
     and decays.
    '''

    def __init__(self, n=0, device=torch.device('cpu')):
        self.device = device
        self.n = n    # Number of nodes

        # Node activities
        self.x = []          # state of the node
        self.dxdt = []       # derivatives of nodes (wrt time)
        # bias (only used for error nodes)
        self.bias = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.tau = 0.1       # time constant
        self.batchsize = 0
        self.idx = -99       # index (for use in PCNetwork class)

        self.type = 'value'
        self.clamped = False

        self.x_decay = 0.    # Activity decay

        # Probe variables
        self.probe_on = False
        self.x_history = []


    #=======
    # Setting behaviours
    def SetTau(self, tau):
        self.tau = tau

    def SetType(self, ltype):
        self.type = ltype


    def Clamped(self, is_clamped):
        '''
         lyr.Clamped(is_clamped)
         Clamps (True) or unclamps (False) the value stored in the layer.
         If it is clamped, the values are not updated.
        '''
        self.clamped = is_clamped



    #======================================================
    #======================================================
    #======================================================
    #
    # Dynamics
    #
    def RateOfChange(self, current):
        self.dxdt += current

    def Decay(self, t):
        '''
         lyr.Decay()
         Adds the decay term to the right-hand side of the differential
         equation, updating dxdt. The input t is the current time.
        '''
        self.dxdt -= self.x_decay*self.x + self.bias

    def Step(self, dt=0.001):
        if not self.clamped:
            self.x += self.dxdt*dt/self.tau
        self.dxdt.zero_()
        if self.probe_on:
            self.x_history.append(deepcopy(self.x.cpu()))




    #=======
    # Allocate and initialization
    def Allocate(self, batchsize=1):
        if batchsize!=self.batchsize:
            self.batchsize = batchsize
            del self.x, self.dxdt, self.x_history
            self.x_history = []
            self.x = torch.zeros(batchsize, self.n, dtype=torch.float32, device=self.device)
            self.dxdt = torch.zeros(batchsize, self.n, dtype=torch.float32, device=self.device)

    def Reset(self, random=0.):
        self.ClearHistory()
        self.ResetState(random=random)

    def ResetState(self, random=0.):
        '''
         lyr.ResetState(random=0.)
         Resets the nodes of the layer to Gaussian random
         values with standard deviation of 'random'.
        '''
        if self.batchsize==0:
            return
        if random==0.:
            self.x.zero_()
        else:
            self.x = torch.randn(self.x.shape[0], self.x.shape[1], dtype=torch.float32, device=self.device) * random
        self.dxdt.zero_()

    def ClearHistory(self):
        del self.x_history
        self.x_history = []

    def SetDecay(self, lam):
        '''
         lyr.SetDecay(lam)
         Sets the decay of the layer to lam, whether it is a value layer
         or an error layer.
         Inputs:
           lam   a scalar
        '''
        self.x_decay = lam

    def SetActivityDecay(self, lam):
        '''
         lyr.SetActivityDecay(lam)
         Sets the activity decay to lam, but only on value layers.
         This call does nothing for error layers.
         Inputs:
           lam   a scalar
        '''
        if self.type=='value':
            self.x_decay = lam

    def SetState(self, x):
        #self.x = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.x = x.detach().clone()

    def SetBias(self, x=None, random=0.):
        if x!=None:
            self.bias = x.clone().detach()
        else:
            self.bias = torch.randn(self.n, dtype=torch.float32, device=self.device) * random

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
        xh = torch.stack(self.x_history, dim=0)
        if self.probe_on:
            for i in idx:
                plt.plot(t_history, xh[:,i,:])





# end
