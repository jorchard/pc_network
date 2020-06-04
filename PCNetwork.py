
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import PCLayer
import PCConnection

dtype = torch.float32
# if torch.cuda.is_available():
#     device = torch.device("cuda:5") # Uncomment this to run on GPU
# else:
#     device = torch.device("cpu")
global device

class PCNetwork():

    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.lyr = []       # list of layers
        self.n_layers = 0   # number of layers
        self.con = []       # list of connections
        self.batchsize = 0  # size of batch
        self.t_history = [] # for recording (probes)
        self.t = 0.         # current simulation time
        self.probe_on = False # becomes True if at least one layer has a probe on
        self.blackout = 0.  # How long to wait during a hold before
                            # turning learning on.
        self.learning_on = False # Used to control the blackout period

        '''
        Weight decay
        The weight decay rate can change as training progresses. It is
        implemented using 2 numbers:
          - init_wd: initial weight decay rate, and
          - drop_wd: fractional drop per epoch
        Then, if e is the number of epochs, the weight decay is
          init_wd * np.exp( np.log(1-drop_wd)*e )
        '''
        self.init_wd = 0.
        self.drop_wd = 0.
        #self.WeightDecay = (lambda e: init_wd*np.exp(np.log(1.-drop_wd)*e))


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
    def Learn(self, data, T, dt=0.001, epochs=5):
        '''
         net.Learn(data, T, dt=0.001, epochs=5)
         Perform learning on the network.

         Inputs:
           data   a DataLoader object, or a list containing two tensors,
                  where data[0] are inputs, and data[1] are targets
           T      how long to hold each sample (in seconds)
           dt     step size (in seconds)
           epochs number of epochs
           blackout determines how long to wait during a hold before
                  turning learning on.
        '''
        #self.Learning(True)
        self.lyr[0].Clamped(True)
        self.lyr[-1].Clamped(True)

        for k in range(epochs):
            #batches = MakeBatches(x, t, batchsize=batchsize, shuffle=True)
            if type(data) in (list, ):
                data = [data]
            n_batches = len(data)
            print('Epoch: '+str(k)+' weight decay = '+str(self.CurrentWeightDecay(k)))
            for batch_idx,b in enumerate(data):
                epoch_time = k + batch_idx/n_batches
                #print(epoch_time, self.WeightDecay(epoch_time))
                self.SetWeightDecay(self.CurrentWeightDecay(epoch_time))
                self.ResetState(random=0.5)
                self.SetInput(b[0])
                self.SetOutput(b[1])
                self.Run(T, dt=0.001)



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

        self.learning_on = False
        t_start = self.t
        while self.t < t_start + T:
            # Learning blackout
            if self.t-t_start > self.blackout:
                self.learning_on = True
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

        # Delivers current across connections
        # and overwrites dMdt and dWdt if a connection's learning is on.
        for c in self.con:
            c.RateOfChange()
        # Apply activity decay (where appropriate)
        for l in self.lyr:
            l.Decay(self.t)


    def Step(self, dt=0.001):
        # Increment the state of each layer
        for l in self.lyr:
            l.Step(dt=dt)
        # Increment the connection weights (potentially)
        if self.learning_on:
            for c in self.con:
                c.Step(dt=dt)


    #=======
    # Setting behaviours
    def Learning(self, learning_on):
        '''
         net.Learning(learning_on)
         Turn learning on (True) or off (False) for all 'general' connections.
        '''
        self.learning_on = learning_on
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

    def SetGamma(self, gamma):
        for c in self.con:
            c.SetGamma(gamma)

    def SetActivityDecay(self, lam):
        for l in self.lyr:
            l.SetActivityDecay(lam) # does nothing on error layers

    def SetDynamicWeightDecay(self, init_wd, drop_wd):
        '''
         net.SetDynamicWeightDecay(init_wd, drop_wd)
         Sets the weight decay parameters.
         The weight decay rate can change as training progresses. It is
         implemented using 2 numbers:
           - init_wd: initial weight decay rate, and
           - drop_wd: fractional drop per epoch
         Then, if e is the number of epochs, the weight decay is
           init_wd * np.exp( np.log(1-drop_wd)*e )
        '''
        self.init_wd = init_wd
        self.drop_wd = drop_wd
        #self.WeightDecay = (lambda e: init_wd*np.exp(np.log(1.-drop_wd)*e))

    def CurrentWeightDecay(self, fe):
        return self.init_wd*np.exp(np.log(1.-self.drop_wd)*fe)

    def SetWeightDecay(self, lam):
        for c in self.con:
            c.SetWeightDecay(lam)

    def SetRepelSmallWeights(self, rho):
        for c in self.con:
            c.SetRepelSmallWeights(rho)

    def SetBlackout(self, t):
        '''
         net.SetBlackout(t)
         Sets the blackout period for learning. The blackout period is
         how long to wait during a old before turning learning on.
        '''
        self.blackout = t


    #=======
    # Building utilities
    def AddLayer(self, l):
        '''
         net.AddLayer(l)
         Adds a layer to the network 'net', where l is a network object.
        '''
        self.lyr.append(l)
        self.n_layers = len(self.lyr)
        l.idx = self.n_layers-1


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
            c = PCConnection.DenseConnection(v=self.lyr[v_idx], e=self.lyr[e_idx], sym=sym, act_text=act_text, device=self.device)
        elif type=='1to1':
            c = PCConnection.DenseConnection(v=self.lyr[v_idx], e=self.lyr[e_idx], type=type, act_text='identity', device=self.device)
            c.SetIdentity()
            self.lyr[e_idx].SetBias(random=1.)

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
        self.ClearHistory()
        self.ResetState(random=random)

    def ClearHistory(self):
        del self.t_history
        self.t_history = []
        self.t = 0.
        for l in self.lyr:
            l.ClearHistory()

    def ResetState(self, random=0.):
        for l in self.lyr:
            l.ResetState(random=random)

    #=======
    # Utilities
    def Probe(self, probe_on=True):
        '''
         net.Probe(probe_on=True)
         Turns on (True) or off (False) data history recording for
         all the layers in the network.
        '''
        for l in self.lyr:
            l.Probe(probe_on)

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





# end
