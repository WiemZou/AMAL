# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from datamaestro import prepare_dataset 


class Context:
    """Very simplified context object"""
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class Linear1(Function):

    @staticmethod
    def forward(ctx,x,w,b):
        input = [x,w,b]
        ctx.save_for_backward(input)
        return torch.dot(x,w) + b

    @staticmethod
    def backward(ctx,grad_output):
        x,w = ctx.saved_tensors
        Lx = torch.dot(grad_output,w)
        Lw = torch.dot(grad_output,x)
        Lb = grad_output
        return Lx, Lw, Lb


## Exemple d'implementation de fonction a 2 entrÃ©es
class MSE(Function):

    @staticmethod
    def forward(ctx,y,y_c):
        ctx.save_for_backward(y,y_c)
        return 

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrÃ©es
        x,w = ctx.saved_tensors
        return None, None

## Pour utiliser la fonction 
mafonction = MaFonction()
ctx = Context()
output = mafonction.forward(ctx,x,w)
mafonction_grad = mafonction.backward(ctx,1)

## Pour tester le gradient 
mafonction_check = MaFonction.apply
x = torch.randn(10,5,requires_grad=True,dtype=torch.float64)
w = torch.randn(1,5,requires_grad=True,dtype=torch.float64)
torch.autograd.gradcheck(mafonction_check,(x,w))

## Pour telecharger le dataset Boston
ds=prepare_dataset("edu.uci.boston")
fields, data =ds.files.data() 

#voir solutions sur le tuto de pytorch comparer resultats
