import numpy as np
import math
#加速梯度法可以看作梯度下降法与镜像梯度法的混合
def calc_gradient(func, x, delta_x):
    val_at_x = func(x)
    val_at_next = func(x + delta_x)
    return (val_at_next - val_at_x) / delta_x
    #计算梯度的函数
#Nesterov加速梯度下降的具体实现
def nesterov_descent(func, L, dimension, init_x=None, numerical_gradient=True, delta_x=0.0005, gradient_func=None,epsilon=None):
    #初始化参数
    assert delta_x > 0

    if (init_x is None):
        x = np.zeros(dimension) 
    else:
        x = init_x

    if (epsilon is None):
        epsilon = 0.05

    lambda_prev = 0
    lambda_curr = 1
    gamma = 1
    y_prev = x
    alpha = 0.05 / (2 * L)

    if numerical_gradient:
        gradient = calc_gradient(func, x, delta_x)
    else:
        gradient = gradient_func(x)

    while np.linalg.norm(gradient) >= epsilon:
        y_curr = x - alpha * gradient
        x = (1 - gamma) * y_curr + gamma * y_prev
        y_prev = y_curr
        #对参数更新求得新的迭代点
        lambda_tmp = lambda_curr
        lambda_curr = (1 + math.sqrt(1 + 4 * lambda_prev * lambda_prev)) / 2
        lambda_prev = lambda_tmp

        gamma = (1 - lambda_prev) / lambda_curr

        if numerical_gradient:
            gradient = calc_gradient(func, x, delta_x)
        else:
            gradient = gradient_func(x)

    return x