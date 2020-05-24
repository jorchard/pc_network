
import numpy as np
import torch
import pickle
import PCLayer
from copy import deepcopy
from abc import abstractmethod



dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device("cuda:0") # Uncomment this to run on GPU
else:
    device = torch.device("cpu")


class PCConnection():

    def __init__(self, v=None, e=None, lower_layer=None, act_text='identity'):
        '''
         con = PCConnection(v=None, e=None, act_text='identity')
         Creates a PCConnection object from PCLayers 'v' to 'e'.

         Inputs:
           v      a PCLayer object, or None
           e      a PCLayer object, or None
           lower_layer is either None, or the same as v or e. It indicates which
                  layer is closer to the input side. If it is None, then the
                  layer with the lower index is chosen.
           act_text    one of: 'identity', 'logistic'
        '''
        if isinstance(v, PCLayer.PCLayer) and isinstance(e, PCLayer.PCLayer):
            self.v = v
            self.e = e
            self.v_idx = v.idx
            self.e_idx = e.idx
        else:
            self.v = []
            self.e = []
            self.v_idx = -99
            self.e_idx = -99
        self.v.SetType('value')
        self.e.SetType('error')
        self.learning_on = False
        self.gamma = 0.1       # Learning time constant
        # This next part sets M_sign to account for the fact that the
        # connections going UP the network are exicitatory, and the
        # connections going down are inhibitory. M_sign is the multiplier
        # for the M direction.
        if lower_layer==None:
            if self.v_idx<self.e_idx:    # if (v) --M--> (e)
                self.M_sign = 1.
            elif self.e_idx<self.v_idx:  # if (e) <--M-- (v)
                self.M_sign = -1.
            else:
                self.M_sign = 0.
        else:
            if lower_layer==self.v_idx:  # if (v) --M--> (e)
                self.M_sign = 1.
            else:                        # if (e) <--M-- (v)
                self.M_sign = -1.
        self.SetActivationFunction(act_text)


    def SetActivationFunction(self, act_text):
        '''
         conn.SetActivationFunction(act_text)
         Sets the activation function for the connection.
         The activation function is only applied to the adjacent v layer.
        '''
        self.act_text = act_text
        if self.act_text=='logistic':
            self.sigma = self.Logistic
            self.sigma_p = self.Logistic_p
        elif self.act_text=='identity':
            self.sigma = self.Identity
            self.sigma_p = self.Identity_p
        elif self.act_text=='tanh':
            self.sigma = self.Tanh
            self.sigma_p = self.Tanh_p

    def SetGamma(self, gamma):
        self.gamma = gamma


    #=======
    # Dynamics
    def RateOfChange(self):
        '''
         con.RateOfChange(t)
         Adds the between-layer connection terms to the derivatives of
         v and e.
        '''
        self.CurrentTo_e()
        self.CurrentTo_v()
        if True: #self.learning_on:
            self.RateOfChange_Weights()

    @abstractmethod
    def CurrentTo_e(self):
        '''
         con.CurrentTo_e()
         Updates the input current to the connected e layer.
        '''
        pass

    @abstractmethod
    def CurrentTo_v(self):
        '''
         con.CurrentTo_e()
         Updates the input current to the connected e layer.
        '''
        pass

    def IncrementToWeights(self):
        '''
         con.IncrementToWeights()
         Updates the derivatives of the weights.
        '''
        pass

    def Step(self, dt=0.001):
        pass


    #=======
    # Setting behaviours
    def Learning(self, learning_on):
        pass



    #=======
    # Activation functions
    def Logistic(self):
        '''
         conn.Logistic()
         Applies the logistic function to the values in layer v.
         Returns a tensor the same size as v.x.
        '''
        return 1. / ( 1. + torch.exp(-self.v.x) )
    def Logistic_p(self):
        '''
         conn.Logistic_p()
         Computes the derivative of the logistic function of the values
         in layer v.
         Returns a tensor the same size as v.x.
        '''
        h = self.Logistic()
        return h * ( 1. - h )

    def Tanh(self):
        '''
         conn.Tanh()
         Applies the tanh function to the values in layer v.
         Returns a tensor the same size as v.x.
        '''
        return torch.tanh(self.v.x)
    def Tanh_p(self):
        '''
         conn.Tanh_p()
         Computes the derivative of the tanh function of the values
         in layer v.
         Returns a tensor the same size as v.x.
        '''
        return 1. - torch.pow(torch.tanh(self.v.x), 2)

    def Identity(self):
        return self.v.x
    def Identity_p(self):
        return torch.ones_like(self.v.x)



class DenseConnection(PCConnection):

    def __init__(self, v=None, e=None, sym=False, type='general', act_text='identity'):
        PCConnection.__init__(self, v=v, e=e, act_text=act_text)

        self.type = type
        if self.type=='1to1':
            self.learning_on = False
        else:
            self.learning_on = True

        # Create weight matrices
        self.sym = sym
        self.M = torch.randn(self.v.n, self.e.n, dtype=torch.float32, device=device)
        if self.sym:
            self.W = deepcopy(self.M.transpose(1,0))
        else:
            self.W = torch.randn(self.e.n, self.v.n, dtype=torch.float32, device=device)

        self.dMdt = torch.zeros_like(self.M)
        self.dWdt = torch.zeros_like(self.W)

        self.M_decay = 0.
        self.W_decay = 0.



    #=======
    # Dynamics
    def CurrentTo_v(self):
        self.v.RateOfChange( -self.M_sign * self.e.x@self.W * self.sigma_p() )

    def CurrentTo_e(self):
        self.e.RateOfChange( self.M_sign * self.sigma()@self.M )

    def RateOfChange_Weights(self):
        '''
         densecon.RateOfChange_Weights()
         Sets the derivative of the weights (w.r.t. time), including decay.
        '''
        sigmax_times_e = ( self.sigma().transpose(1,0) @ self.e.x ) / self.v.batchsize
        self.dMdt = -self.M_sign * sigmax_times_e - self.M_decay*self.M
        self.dWdt = -self.M_sign * sigmax_times_e.transpose(1,0) - self.W_decay*self.W

    def Step(self, dt=0.001):
        if self.learning_on:
            self.M += self.dMdt*dt/self.gamma
            self.W += self.dWdt*dt/self.gamma

    #=======
    # Setting behaviours
    def Learning(self, learning_on):
        if self.type=='general':
            self.learning_on = learning_on
        else:
            self.learning_on = False

    def SetWeightDecay(self, lam):
        self.M_decay = lam
        self.W_decay = lam


    #=======
    # Weight matrices
    def SetIdentity(self, mult=1., random=0.):
        '''
         con.SetIdentity(mult=1.)
         Sets the connection weights to mult times the identity matrix.
        '''
        assert (self.v.n==self.e.n), 'Cannot use identity matrix: Number of nodes do not match'
        self.M = mult*torch.eye(self.v.n) + torch.randn_like(self.M)*random
        if self.sym:
            self.W = deepcopy(self.M)
        else:
            self.W = mult*torch.eye(self.v.n) + torch.randn_like(self.W)*random

    def SetRandom(self, random=1.):
        if self.type=='general':
            self.M = torch.randn(self.v.n, self.e.n, dtype=torch.float32, device=device) * random
            if self.sym:
                self.W = self.M.transpose(1,0).clone().detach()
            else:
                self.W = torch.randn(self.e.n, self.v.n, dtype=torch.float32, device=device) * random
# end
