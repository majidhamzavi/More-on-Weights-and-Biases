#Pseudoinverse
import numpy as np
from scipy.special import logit


# Activation functions
def ReLU(x):
    return x * (x > 0)

def Sig(x):
    return 1/(1+np.exp(-x))

def onehot(X):
    T = np.zeros((X.shape[0],np.max(X)+1))
    T[np.arange(len(X)),X] = 1 #Set T[i,X[i]] to 1
    return T

# feed forward function
def ff(w0,w1,w2,b0,b1,b2,x):
    # First Hidden calculations
    h0 = x.dot(w0)+ b0 
    h0 = ReLU(h0)
 
  # Second Hidden calculations
    h1 = h0.dot(w1) + b1
    h1 = ReLU(h1)

  # output calculations
    ff = h1.dot(w2) + b2
    ff = Sig(ff)
    return ff


# data preprocessing
    # train data
xtrain = np.loadtxt('xtrain.txt', delimiter=',')
xtrain /= 255
ytrain = np.loadtxt('ytrain.txt', delimiter=',').astype(int)
ytrain = onehot(ytrain)
   #test data
xtest = np.loadtxt('xtest.txt', delimiter=',')
xtest /= 255
ytest = np.loadtxt('ytest.txt', delimiter=',').astype(int)

# randomly-chosen set of weights and biase
s = 0.2
W0 = (np.random.rand(784,512)-0.5)*s
W1 = (np.random.rand(512,512)-0.5)*s

B0 = (np.random.rand(512,)-0.5)*s
B1 = (np.random.rand(512,)-0.5)*s


# WieghtsPseudoinvers
H0 = ReLU(xtrain.dot(W0) + B0)
H1 = ReLU(H0.dot(W1) + B1)
ytrain_transformed = ytrain * 0.9 + 0.05
Y_l = logit(ytrain_transformed)
B2 = np.mean(Y_l,0)
W2 = np.linalg.pinv(H1).dot(Y_l - B2)


# test 
Output = ff(W0,W1,W2,B0,B1,B2,xtest)
Output = np.argmax(Output, axis = 1)


#ytrain1 = np.loadtxt('ytrain.txt', delimiter=',').astype(int)
accuracy = ((np.count_nonzero(Output==ytest))/10000)*100       
        
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix        
cm = confusion_matrix(Output, ytest)
print(accuracy)


