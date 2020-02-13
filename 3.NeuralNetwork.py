import numpy as np
import math as math

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

# TBD