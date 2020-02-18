import numpy as np
import math as math
import matplotlib.pylab as plt

# 3 Neural network

# 3.1 from Perceptron to Neural network

# review

# y= 0 ( b+w1x1+w2x2 <= 0 )
# y= 1 ( b+w1x1+w2x2 > 0 )
# b: bias
# wx wight

# means...

# 1 * b +
# x1*w1 + => y
# x2*w2 +

# 3.2 Activation function

# y=h( b+w1x1+w2x2 )
# h(x)= 0(x<=0),1(x>0) # called "Activation function" "JP:活性化関数"

# in detail...

# 1 * b +
# x1*w1 + => a => h() => y
# x2*w2 +

# a:sum of input(s)
# h():activation function
# output

# neuron equals node( in this project )

# 3.1 Sigmoid function

# h(x)=1/(  1 + exp(-x) )
# e:Napier's constants

# Let's try as I want

def doSigmoid(x):
    y=1.0/(  1.0 + math.e**(-x) )
    return y


for i in [1,2,3,4,5,6,7,8,9,10]:
    val=i*(-1)
    print( "Sigmoid("+str(val)+")=>"+str(doSigmoid(val)) )

for i in [1,2,3,4,5,6,7,8,9,10]:
    val=i*(+1)
    print( "Sigmoid("+str(val)+")=>"+str(doSigmoid(val)) )

for i in [1,2,3,4,5,6,7,8,9,10]:
    val=i*(-2)
    print( "Sigmoid("+str(val)+")=>"+str(doSigmoid(val)) )

for i in [1,2,3,4,5,6,7,8,9,10]:
    val=i*(+2)
    print( "Sigmoid("+str(val)+")=>"+str(doSigmoid(val)) )

# 3.2.2 Step function

# Step function is simple function which returns 1 when the value is over zero and returns 0 when the value equals or less than zero
def step_function(x): # 1 dimension
    if x>0:
        return 1
    else:
        return 0

# However step function can NOT handle N dimensions such as array and
# So define as below
def step_function(x): # 2 or more dimensions
    y=x>0
    #print("DBG:"+"x="+str(x)+",y="+str(y))
    return np.array(x>0, dtype=np.int)

# execute test
x=np.array([-1.0,1.0,2.0])
print(x)
y=step_function(x)
print( y )

# 3.2.3 Graph of Step Function

x=np.arange(-5.0, 5.0, 0.1)
y=step_function(x)

print(x)
print( y )

plt.plot(x,y)
plt.ylim(-0.1,1.1)
# plt.show()

# 3.2.4 Sigmoid function

def sigmoid(x):
    return 1/(1+np.exp(-x)) # exp -> exponential 'e^(-x)'

x=np.array([-1.0,1.0,2.0])
s=sigmoid(x)
print(x)
print(s)

# Graph of sigmoid
x=np.arange(-5.0, 5.0, 0.1)
y=sigmoid(x)

print(x)
print( y )

plt.plot(x,y)
plt.ylim(-0.1,1.1)
#plt.show()

# 3.2.6 Nonlinear function
#   Activation function MUST be Nonlinear function such as step/sigmouid

# 3.2.7 ReLU function, Rectified Linear Unit

# ReLU is
# h(x)=x(x>0)
#     =0(x<=0)

def relu(x):
    return np.maximum(0,x)

x=np.arange(-5.0, 5.0, 0.1)
y=relu(x)

print(x)
print( y )

plt.plot(x,y)
plt.ylim(-0.1,1.1)
#plt.show()

# 3.3 multi dimensions
# 3.3.1 multi dimensions array

# 1 dimension

A=np.array([1,2,3,4])
print(A)
B=np.ndim(A)
print("Dimension="+str(B))

print("Shaped:"+str(A.shape)) # displayed as tuple
print("Shaped[0]:"+str(A.shape[0]))

# 2 dimensions

B=np.array([[1,2],[3,4],[5,6]])
print("B:"+str(B))
print("np.ndim(B):"+str(np.ndim(B)))
print("B.shape:"+str(B.shape))


# 3.3.2 matrix multiplication

A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])

M=np.dot(A,B)

print("A:")
print(str(A))
print("B:")
print(str(B))
print("matrix multiplication:")
print(str(M))


A=np.array([[1,2,3],[4,5,6]])
B=np.array([[1,2],[3,4],[5,6]])

M=np.dot(A,B)

print("A:")
print(str(A))
print("B:")
print(str(B))
print("matrix multiplication:")
print(str(M))

# Notice array(X,Y) array(Z,K)
#                ^        ^
# Y must equal to Z to calculate N dmensions arrangement

# So far, common math
# by here 2020.02.14 15:37

C=np.array([[1,2],[3,4]])
print(C.shape)
print(A.shape)
#np.dot(A,C)

'''
Traceback (most recent call last):
  File "*****/venv/3.NeuralNetwork.py", line 202, in <module>
    np.dot(A,C)
  File "<__array_function__ internals>", line 6, in dot
ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)
                      ^       ^ 
                      3  <>   2
'''

A=np.array([[1,2],[3,4],[5,6]])   # 3*2
B=np.array([[1,2,3,4],[5,6,7,8]]) # 2*4

# 3*2  *  2*4  =  3*4
C=np.dot(A,B)
print(C)
