
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
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
        self.probe_on = False # becomes True if at least one layer has a probe on


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
    def Learn(self, dl, T, dt=0.001, epochs=5):
        '''
         net.Learn(dl, T, dt=0.001, epochs=5)
         Perform learning on the network.

         Inputs:
           dl     a DataLoader object
           T      how long to hold each sample (in seconds)
           dt     step size (in seconds)
           epochs number of epochs
        '''
        self.Learning(True)
        self.lyr[0].Clamped(True)
        self.lyr[-1].Clamped(True)

        for k in range(epochs):
            #batches = MakeBatches(x, t, batchsize=batchsize, shuffle=True)
            for b in iter(dl):
                self.SetInput(b[0])
                self.SetOutput(b[1])
                self.Run(T, dt=0.001)
            print('Epoch: '+str(k))



    def Predict(self, x, T, dt=0.001):
        self.Learning(False)
        self.lyr[0].Clamped(True)
        self.lyr[-1].Clamped(False)
        self.SetInput(x)
        self.Run(T, dt=dt)
        return self.lyr[-1].x

    def Generate(self, t, T, dt=0.001):
        self.Learning(False)
        self.lyr[0].Clamped(False)
        self.lyr[-1].Clamped(True)
        self.SetOutput(t)
        self.Run(T, dt=dt)
        return self.lyr[0].x


    def Run(self, T, dt=0.001):
        '''
         net.Run(T, dt=0.001)
         Simulates the network for T seconds using a time step of dt.
         The network state is continued from the previous run, if one
         exists. Calling net.Reset() removes the previous run.
        '''
        self.probe_on = False
        for l in self.lyr:
            self.probe_on = self.probe_on or l.probe_on

        t_start = self.t
        while self.t < t_start + T:
            self.RateOfChange()
            self.Step(dt=dt)
            self.t += dt
            if self.probe_on:
                self.t_history.append(self.t)

    def RateOfChange(self):
        '''
         net.RateOfChange()
         Updates the input currents to all nodes in the network
        '''
        for l in self.lyr:
            l.dxdt.zero_()

        # Deliver current across connections
        for c in self.con:
            c.RateOfChange()
        # Apply activity decay (where appropriate)
        for l in self.lyr:
            l.Decay()


    def Step(self, dt=0.001):
        # Increment the state of each layer
        for l in self.lyr:
            l.Step(dt=dt)
        # Increment the connection weights (potentially)
        for c in self.con:
            c.Step(dt=dt)


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

    def SetOutput(self, t):
        self.Allocate(t)
        self.lyr[-1].SetState(t)

    def SetTau(self, tau):
        for l in self.lyr:
            l.SetTau(tau)

    def SetActivityDecay(self, lam):
        for l in self.lyr:
            l.SetActivityDecay(lam) # does nothing on error layers

    def SetWeightDecay(self, lam):
        for c in self.con:
            c.SetWeightDecay(lam)

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
            c = PCConnection.DenseConnection(v=self.lyr[v_idx], e=self.lyr[e_idx], type=type, act_text='identity')
            c.SetIdentity()

        self.lyr[e_idx].SetDecay(1.)  # Set decay of error layer
        
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


    #=======
    # Utilities
    def Plot(self, idx=0):
        '''
         net.Plot(idx=0)
         Plots the time evolution of sample idx in the batch.
         It displays an array of plots, one plot for each layer.

         The plots are in 2 rows, with odd-index layers in the top row,
         and even-index layers in the bottom row.
        '''
        fig = plt.figure(figsize=(10,4), constrained_layout=True)
        n_valnodes = np.ceil(self.n_layers/2.)
        gs = plt.GridSpec(2, int(2*n_valnodes), figure=fig)
        r,c = 0,0
        for l in self.lyr:
            fig.add_subplot(gs[r,c:c+2])
            l.Plot(self.t_history, idx=idx)
            if r==0 and c==0:
                plt.ylabel('v')
            elif r==1 and c==1:
                plt.ylabel('e')
            if r==1:
                plt.xlabel('Time (s)')
            r = (r+1)%2
            c += 1
        return fig








#============================================================
#
# Untility functions
#
#============================================================
def MakeBatches(data_in, data_out, batchsize=10, shuffle=True):
    '''
        batches = MakeBatches(data_in, data_out, batchsize=10, shuffle=True)

        Breaks up the dataset into batches of size batchsize.

        Inputs:
          data_in    is a list of inputs
          data_out   is a list of outputs
          batch size is the number of samples in each batch
          shuffle    shuffle samples first (True)

        Output:
          batches is a list containing batches, where each batch is:
                     [in_batch, out_batch]


        Note: The last batch might be incomplete (smaller than batchsize).
    '''
    N = len(data_in)
    r = range(N)
    if shuffle:
        r = torch.randperm(N)
    batches = []
    for k in range(0, N, batchsize):
        if k+batchsize<=N:
            din = data_in[r[k:k+batchsize]]
            dout = data_out[r[k:k+batchsize]]
        else:
            din = data_in[r[k:]]
            dout = data_out[r[k:]]
        if isinstance(din, (list, tuple)):
            batches.append( [torch.stack(din, dim=0).float().to(device) , torch.stack(dout, dim=0).float().to(device)] )
        else:
            batches.append( [din.float().to(device) , dout.float().to(device)] )
    return batches

# end
