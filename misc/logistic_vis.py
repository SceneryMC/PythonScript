import matplotlib.pyplot as plt

def logistic_map(r, x0, n):
    """
    生成逻辑斯谛映射序列。
    r: 控制参数
    x0: 初值
    n: 迭代次数
    """
    x = [x0]
    for _ in range(n - 1):
        x_next = r * x[-1] * (1 - x[-1])
        x.append(x_next)
    return x

# 设置参数
R_CHAOTIC = 3.9  # 一个会导致混沌的 r 值
ITERATIONS = 50   # 迭代次数

# 两个极其接近的初值
x0_1 = 0.2
x0_2 = 0.2 + 1e-8

# 生成序列
sequence_1 = logistic_map(R_CHAOTIC, x0_1, ITERATIONS)
sequence_2 = logistic_map(R_CHAOTIC, x0_2, ITERATIONS)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(sequence_1, 'bo-', label=f'x0 = {x0_1}')
plt.plot(sequence_2, 'ro-', label=f'x0 = {x0_2}')
plt.title(f'Logistic Map (r = {R_CHAOTIC}) - The Butterfly Effect')
plt.xlabel('Iteration (n)')
plt.ylabel('Value (Xn)')
plt.legend()
plt.grid(True)
plt.show()