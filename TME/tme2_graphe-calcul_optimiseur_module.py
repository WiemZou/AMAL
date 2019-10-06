
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

class Context:
    """Very simplified context object"""
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors

device = 'cpu'
dtype = torch.float
x = torch.randn((1,10),requires_grad=True,dtype=torch.float,device=device)
w = torch.randn((1,10),requires_grad=True,dtype=torch.float,device=device)
#b = torch.randn(1,1,requires_grad=True,dtype=torch.float64)
y = torch.randint(2,size=(1,),dtype=torch.float,device='cpu')

learning_rate = 10e-3
#data, target = load_boston(return_X_y=True)
#X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)



""" for i in range(200):
    y_pred = x.mm(w.T)
    mse = (y_pred-y).pow(2)
    mse.backward()
    print("y_pred :",y_pred," y_true :",y," loss :",mse)
    with torch.no_grad():
        w -= learning_rate*w.grad
        w.grad.zero_() """

""" optim = torch.optim.SGD(params=[w],lr=0.001)

for i in range(50):
    y_pred = x.mm(w.T)
    mse = (y_pred-y).pow(2)
    mse.backward()
    optim.step()
    optim.zero_grad() """

class Wiwi(torch.nn.Module):

    def __init__(self,dim_in,dim_out):
        super(Wiwi,self).__init__()
        self.linear = torch.nn.Linear(dim_in,dim_out)

    def forward(self,x):
        y = self.linear(x).squeeze()
        return y 
    

loss = torch.nn.MSELoss()
learning_rate = 10e-3
model = Wiwi(x.size()[1],x.size()[0])
optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
for i in range(50):
    model.train()
    y_pred = model(x) #equivalent a model.forward(x)
    z = loss(y_pred,y)
    z.backward()
    optim.step()
    optim.zero_grad()

#Warning a voir 
