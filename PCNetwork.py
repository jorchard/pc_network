
import numpy as np
import torch
import pickle
import PCLayer
import PCConnection

dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device("cuda:0") # Uncomment this to run on GPU
else:
    device = torch.device("cpu")


class PCNetwork():

    def __init__(self):
        self.lyr = []       # list of layers
        self.n_layers = 0   # number of layers
        self.con = []       # list of connections
        self.batchsize = 0  # size of batch
        self.t_history = [] # for recording (probes)
        self.t = 0.         # current simulation time



    #=======
    # Save and Load
    def Save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    @classmethod
    def Load(cls, fname):
        with open(fname, 'rb') as fp:
            net = pickle.load(fp)
        return net


    #=======
    # Dynamics
    def RateOfChange(self):
        '''
         net.RateOfChange()
         Updates the input currents to all nodes in the network
        '''
        for c in self.con:
            c.RateOfChange()

        for l in self.lyr:
            l.Decay()


    def Step(self, dt=0.001):
        for l in self.lyr:
            l.Step(dt=dt)
        for c in self.con:
            l.Step(dt=dt)


    #=======
    # Setting behaviours
    def Learning(self, learning_on):
        '''
         net.Learning(learning_on)
         Turn learning on (True) or off (False) for all 'general' connections.
        '''
        for c in self.con:
            c.Learning(learning_on)

    def SetInput(self, x):
        self.Allocate(x)
        self.lyr[0].SetState(x)

    #=======
    # Building utilities
    def AddLayer(self, l):
        '''
         net.AddLayer(l)
         Adds a layer to the network 'net', where l is a network object.
        '''
        self.lyr.append(l)
        self.n_layers = len(self.lyr)
        l.idx = self.n_layers


    def Connect(self, v_idx=None, e_idx=None, type='general', sym=False, act_text='logistic'):
        '''
         net.Connect(v=None, e=None, type='general')
         Creates a connection between the value node v and the error
         node e.
         Inputs:
           v_idx, e_idx  are indices to the network's lyr list
           type is either 'general', or '1to1'
           act_text is one of 'logistic', or 'identity'
        '''
        if type=='general':
            c = PCConnection.DenseConnection(v=self.lyr[v_idx], e=self.lyr[e_idx], sym=sym, act_text=act_text)
        elif type=='1to1':
            c = PCConnection.DenseConnection(v=self.lyr[v_idx], e=self.lyr[e_idx], act_text='identity')
            c.SetIdentity()

        self.con.append(c)


    def Allocate(self, x):
        '''
         net.Allocate(x)

         Allocates vectors for the nodes in all the layers.
         This method makes no guarantees about the values in the nodes.

         Input:
           x can either be the number of samples in a batch, or it can be
             a batch.
        '''
        proposed_batchsize = 1
        if type(x) in (int, float, ):
            proposed_batchsize = x
        else:
            proposed_batchsize = len(x)

        if proposed_batchsize!=self.batchsize:
            self.batchsize = proposed_batchsize
            del self.t_history
            self.t_history = []
            self.t = 0.
            for l in self.lyr:
                l.Allocate(batchsize=proposed_batchsize)


    def Reset(self, random=0.):
        '''
         net.Reset(random=0.)
         Resets the simulation of the network, and erases its history.
         It also resets the values in the nodes to random numbers using
         a Gaussian distribution with mean 0 and std specified by the
         input parameter 'random'.
        '''
        del self.t_history
        self.t_history = []
        self.t = 0.
        for l in self.lyr:
            l.Reset(random=random)





# end
