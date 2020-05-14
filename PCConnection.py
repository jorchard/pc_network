
import numpy as np
import torch
import pickle
import PCLayer
from copy import deepcopy

dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device("cuda:0") # Uncomment this to run on GPU
else:
    device = torch.device("cpu")


class PCConnection():

    def __init__(self, v=None, e=None, act_text='identity'):
        '''
         con = PCConnection(v=None, e=None, act_text='identity')
         Creates a PCConnection object from PCLayers 'v' to 'e'.

         Inputs:
           v      a PCLayer object, or None
           e      a PCLayer object, or None
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


    def RateOfChange(self):
        '''
         con.RateOfChange()
         Adds the between-layer connection terms to the derivatives of
         v and e.
        '''
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

    def Identity(self):
        return self.v.x
    def Identity_p(self):
        return torch.ones_like(self.v.x)



class DenseConnection(PCConnection):

    def __init__(self, v=None, e=None, sym=False, act_text='identity'):
        PCConnection.__init__(self, v=v, e=e, act_text=act_text)

        # Create weight matrices
        self.M = torch.randn(self.v.n, self.e.n, dtype=torch.float32, device=device)
        if sym:
            self.W = deepcopy(self.M.transpose(1,0))
        else:
            self.W = torch.randn(self.e.n, self.v.n, dtype=torch.float32, device=device)


    #=======
    # Weight matrices
    def SetIdentity(self):
        assert (self.v.n==self.e.n), 'Cannot use identity matrix: Number of nodes do not match'
        self.M = torch.eye(self.v.n)
        self.W = torch.eye(self.v.n)




# end
