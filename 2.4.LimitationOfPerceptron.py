import numpy as np

# 2.4 パーセプロトンの限界

# 2.4.1 XOR Gate(排他的論理和：異なる場合のみ1)
# これまでのパーセプトロンでは実装できない！

'''
          x2
          |
    (1,0) +   +(1,1)
          |
----------+---+------ x1
    (0,0) |  (0,1)
          |
          |

y= 0 ( -0.5+x1+x2 <= 0 )
y= 1 ( -0.5+x1+x2 > 0 )

ex) x2 = -x1+0.5 ( Linear equation )

ここまでは直線で（4つの点を）分けることが出来た

XOR は
(0,0)(1,1) => 0
(0,1)(1,0) => 1
なので直線では分けられない

=> 非線形（直線じゃないやつ）

'''

# 2.5 多層パーセプトロン
# ではどうやって実現するか => 層を重ねる(AND/OR/NORの組み合わせ)

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

# 組み合わせれば良いだけ






