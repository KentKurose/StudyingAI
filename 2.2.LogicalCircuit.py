import numpy as np


# 2.2 Simple logical circuit

# 2.3 implementation of perceptron 

# 2.3.1 simple implementation
'''
def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tmp=x1+w1+x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print("AND(0,0)")
print(AND(0,0))

print("AND(0,1)")
print(AND(0,1))

print("AND(1,0)")
print(AND(1,0))

print("AND(1,1)")
print(AND(1,1))
'''

# 2.3.2 Bias

'''
x=np.array([0,1]) #input
w=np.array([0.5,0.5]) #weight
b=-0.7 # バイアス

print(x)
print(w)

tmp=x*w
print(tmp)

sum=np.sum(tmp)
print(sum)

print(sum+b)
'''


# 2.3.3 implematation of AND by bias and weight

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


print("AND(0,0)")
print(AND(0,0))

print("AND(0,1)")
print(AND(0,1))

print("AND(1,0)")
print(AND(1,0))

print("AND(1,1)")
print(AND(1,1))


print("NAND(0,0)")
print(NAND(0,0))

print("NAND(0,1)")
print(NAND(0,1))

print("NAND(1,0)")
print(NAND(1,0))

print("NAND(1,1)")
print(NAND(1,1))


print("OR(0,0)")
print(OR(0,0))

print("OR(0,1)")
print(OR(0,1))

print("OR(1,0)")
print(OR(1,0))

print("OR(1,1)")
print(OR(1,1))
