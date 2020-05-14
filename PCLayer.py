
import numpy as np
import copy
import pickle
import torch

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

        # Probe variables
        self.probe_on = False
        self.x_history = []

    #=======
    # Save and Load
    def Save(self, fp):
        '''
         layer.Save(fp)
         Saves the layer to the file pointed to by fp.
        '''
        pickle.dump(self, fp)

    @classmethod
    def Load(cls, fp):
        '''
         layer = PCLayer.Load(fp)
         Loads a layer from the file pointed to by fp.
        '''
        lyr = pickle.load(fp)
        return lyr


    #=======
    # Allocate and initialization
    def Allocate(self, batchsize=1):
        if batchsize!=self.batchsize:
            self.batchsize = batchsize
            del self.x, self.dxdt, self.x_history
            self.x_history = []
            self.x = torch.zeros(batchsize, self.n, dtype=torch.float32, device=device)

    def Reset(self, random=0.):
        del self.x_history
        self.x_history = []
        if random==0.:
            self.x.zero_()
        else:
            self.x = torch.randn(self.x.shape[0], self.x.shape[1], dtype=torch.float32, device=device) * random

    def Probe(self, bool):
        self.probe_on = bool

    def Record(self):
        if self.probe_on:
            self.x_history.append( np.array(self.v.cpu())[0] )








# end
