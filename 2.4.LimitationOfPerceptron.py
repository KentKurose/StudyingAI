import numpy as np

# 2.4 limitation of perceptron

# 2.4.1 XOR Gate
# cannot implement with perceptron which is created so far

'''
          x2
          |
    (0,1) +   +(1,1)
          |
----------+---+------ x1
    (0,0) |  (1,0)
          |
          |

y= 0 ( -0.5+x1+x2 <= 0 )
y= 1 ( -0.5+x1+x2 > 0 )

ex) x2 = -x1+0.5 ( Linear equation )

So far four points could be devided by linear, straight line

XOR canNOT be devided as figure above
(0,0)(1,1) => 0
(0,1)(1,0) => 1

=> nonlinear, non straight line

'''

# 2.5 multilayer perceptron
# How to implement => add layers, combination of AND/OR/NOR

# XOR

# x1 => NAND  +=> AND
# x2 => OR    +

def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(w*x)+b

    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(w*x)+b

    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sum(w*x)+b

    if tmp <= 0:
        return 0
    else:
        return 1

# XOR
def xor(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y



print( "xor(0,0)" )
print( xor(0,0) )

print( "xor(0,1)" )
print( xor(0,1) )

print( "xor(1,0)" )
print( xor(1,0) )

print( "xor(1,1)" )
print( xor(1,1) )

print( "got it !!" )

# Just combining makes it!





