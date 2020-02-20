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

'''

If both a and b are 1-D arrays, ## MATHMATICALLY ## it is inner product of vectors (without complex conjugation) and results in scalar.
https://rikeilabo.com/vector-formula-list#i-9
If both a and b are 1-D arrays, ## NumPy.dot     ## it is inner product of vectors (without complex conjugation) and results in scalar.
https://www.sejuku.net/blog/71827

If both a and b are 2-D arrays, ## MATHMATICALLY ## it is inner product of vectors (without complex conjugation) and results in scalar.
https://mathwords.net/gyoretunonaiseki
If both a and b are 2-D arrays, ## NumPy.dot     ## it is inner product of vectors (without complex conjugation) and results in vector.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html

If both a and b are 3-D(or more) arrays, ## MATHMATICALLY ## it is inner product of vectors (without complex conjugation) and results in scalar.
If both a and b are 3-D(or more) arrays, ## NumPy.dot     ## it is inner product of vectors (without complex conjugation) and results in tensor.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul
also mentions that 
If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:
'''

# 2-D arrays: results of these three are the same

print("2-D arrays: results of these three are the same")

A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])

print("A")
print(A)
print("B")
print(B)

print("np.dot(A,B)")
print(np.dot(A,B))

print("np.matmul(A,B)")
print(np.matmul(A,B))

print("A@B")
print(A@B)

print("np.sum(A@B)")
print(np.sum(A@B))

# 0-D arrays: results of these three are the same
#  but results in sclar using dot, and results in array with others
print("0-D arrays: results of these three are the same")

A=np.array([1])
B=np.array([2])

print("A")
print(A)
print("B")
print(B)

print("np.dot(A,B)")
print(np.dot(A,B))

print("np.multiply(A,B)")
print(np.multiply(A,B))

print("A*B")
print(A*B)

print("A@B")
print(A@B)

print("np.matmul(A,B)")
print(np.matmul(A,B))

print("np.sum(np.dot(A,B))")
print(np.sum(np.dot(A,B)))
print("np.sumnp.multiply(A,B)")
print(np.sum(np.multiply(A,B)))
print("np.sum(A*B)")
print(np.sum(A*B))
print("np.sum(A@B)")
print(np.sum(A@B))

# To results in scalar, can use np.sum(np.dot(A,B)) but sometimes not preffered
# Preffered functions are
# 0-D: numpy.multiply  or    *
# 1-D: numpy.dot
# 2-D: numpy.matmul    or  a @ b

# 3.3.3 matrix multiplication for Neural network

# in this sample
#   Activation function and Bias are are omitted.

print("3.3.3 matrix multiplication for Neural network")

X=np.array([1,2])
print("X:"+str(X))
print("X.shape:"+str(X.shape))

# results in X.shape:(2,)
# I think (1,2) is the right, but I also guess omitted description is used

# To check output
'''
T=np.array([[1,2],[1,2],[1,2]])
print("T.shape:"+str(T.shape))
'''
#  T.shape:(3, 2)

W=np.array([[1,3,5],[2,4,6]])
print("W:"+str(W))
print("W.shape:"+str(W.shape))

Y=np.dot(X,W)
print("Y:"+str(Y))

# this means 2 imputs lead 3 output as petterns
'''

           ->  y1
x1 ->  
           ->  y2
x2 ->
           ->  y3

'''
# 3.4 Three-Layer Neural network




