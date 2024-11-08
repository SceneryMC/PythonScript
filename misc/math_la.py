from sympy.matrices.normalforms import smith_normal_form
import sympy as sp

# 定义符号变量 x
x = sp.Symbol('x')

# 定义多项式矩阵，元素为关于 x 的多项式，系数为实数
A = sp.Matrix([
    [-x + 1, 2 * x - 1, x],
    [x, x ** 2, -x],
    [x ** 2 + 1, x ** 2 + x - 1, -x ** 2],
])

B = sp.Matrix([
    [x ** 2 - 1, 0],
    [0, (x - 1) ** 3],
])

C = sp.Matrix([
    [x ** 3 * (x - 2), 0, 0],
    [0, x * (x + 1), 0],
    [0, 0, x * (x - 2) ** 2],
])

# 计算Smith标准形
domain = sp.QQ[x] #  sp.RR 是实数域，RR[x] 是实数系数的多项式环。虽然实数域是一个域，RR[x] 应该是一个主理想整环，但在某些情况下，sympy 的实现可能会对其处理不够完善，导致不支持某些操作。
SNF = smith_normal_form(C, domain=domain).applyfunc(lambda y: sp.factor(y))

# 输出Smith标准形
print("Smith标准形:")
sp.pprint(SNF)

# M = sp.Matrix([
#     [2, 1, 0, -1],
#     [0, 2, 0, 1],
#     [0, 0, 2, 1],
#     [0, 0, 0, 2],
# ])

# x, y, z = sp.symbols('x y z')
#
# M = sp.Matrix([
#     [2, 4, 2, 2],
#     [1, 2, 1, 1],
#     [1, 2, 1, 1],
#     [1, 2, 1, 1],
# ])
#
# P, J = M.jordan_form()
# sp.pprint(P)
# sp.pprint(J)
from sympy import im
I = sp.I

M = sp.Matrix([
    [1, 0, 2*I],
    [0, 3, 0],
    [-2*I, 0, 1]
])

P, D = M.diagonalize(reals_only=False)
N = sp.Matrix([
    [1 / sp.sqrt(2), 0, 0],
    [0, 1, 0],
    [0, 0, 1 / sp.sqrt(2)],
])
P = P @ N
sp.pprint(P)
sp.pprint(P @ P.H)
sp.pprint(D)
sp.pprint(P.H @ M @ P)

v = sp.Matrix([1,1,1,1]).transpose()
a1 = sp.Matrix([1,0,I,0]).transpose()
a2 = sp.Matrix([1,1,0,0]).transpose()

# Gram-Schmidt 过程
# 首先保持 a1
u1 = a1

# 正交化 a2
u2 = a2 - ((u1.dot(a2, hermitian=True)) / u1.dot(u1, hermitian=True)) * u1
sp.pprint(u2)

# 计算 v 在正交基上的投影
proj_u1_v = (v.dot(u1, hermitian=True) / u1.dot(u1, hermitian=True)) * u1
proj_u2_v = (v.dot(u2, hermitian=True) / u2.dot(u2, hermitian=True)) * u2

sp.pprint(proj_u1_v)
sp.pprint(proj_u2_v)

# 计算总的投影
proj_span = proj_u1_v + proj_u2_v

# 输出结果
print("向量 v 投影到 span(a1, a2) 上的投影:")
sp.pprint(proj_span.applyfunc(lambda y: sp.simplify(y)))
sp.pprint(a1 * 2 / 3 * I + a2 * (1 - I / 3))

a,b,c,d = sp.symbols('a b c d')
A = sp.Matrix([
    [a, b, c, d],
    [b, -a, d, -c],
    [c, -d, -a, b],
    [d, c, -b, -a],
])
sp.pprint(A.inv())
sp.pprint(A @ A.transpose())
