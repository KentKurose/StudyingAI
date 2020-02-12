import numpy as np

# 1.5.2 Numpy配列の生成

x=np.array([1.0,2.0,3.0])
#print(x)
#type(x)

#print(type(x))

# 1.5.3 Numpyの算術計算

x=np.array([1.0,2.0,3.0])
y=np.array([2.0,4.0,6.0])

#print( x+y )
#print( x-y )
#print( x*y )
#print( x/y )

x=np.array([1.0,2.0,3.0])
#print( x/2.0 )

# 1.5.4 NumpyのN次元配列
A=np.array([ [1,2] , [3,4] ])
#print(A)

# 形状
#print(A.shape)
# 型
#print(A.dtype)

# 算術計算


B=np.array([[3,6],[0,6]])

#print("----------- Original Values ")
#print(A)
#print(B)
#print("----------- Adds")
#print(A+B)
#print("----------- Times")
#print(A*B)
#print("----------- 10 times(broadcast)")
#print(A*10)

# 1 Demnsion => Array
# 2 Demnsion => Vector
# 3 or more Demnsion => Tensor

# 1.5.5 Broadcast

A=np.array([[1,2],[3,4]])
B=np.array([10,20])

#print("----------- Original Values ")
#print(A)
#print(B)
#print("----------- Adds")
#print(A+B)
#print("----------- Times")
#print(A*B)

# 1.5.6 Access to 要素

X=np.array([[51,55],[14,19],[0,4]])

'''
print("X")
print(X)
print("X[0]")
print(X[0])
print("X[1]")
print(X[1])
print("X[2]")
print(X[2])
print("X[0,0]")
print(X[0,0])
print("X[1,0]")
print(X[1,0])
print("X[0,1]")
print(X[0,1])
'''

'''
for row in X:
    print(row)
    for val in row:
        print(val)
'''

X=X.flatten() # into Array
print(X)

X[np.array([0,2,4])] # index 0,2,4
print(X[np.array([0,2,4])])

X>15 # Condition
print(X>15)

print(X[X>15])
print(X[X<15])
