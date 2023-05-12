import math

import torch


def neg_logarithmic_mapping(start, end, num_steps):
    # Generate the logarithmic mapping
    interval = end - start
    step_size = interval / num_steps
    x = torch.arange(start, end, step_size)
    c = 1 + 1/math.e ** num_steps
    y = -torch.exp(x - num_steps) + c
    return y


def neg_linear_mapping(start, end, num_steps):
    interval = end - start
    step_size = interval / num_steps
    x = torch.arange(start, end, step_size)
    y = -1/num_steps * x + 1
    return y


def neg_gaussian_mapping(start, end, num_steps, height=1):
    #  rabbit's in a hat
    interval = end - start
    step_size = interval / num_steps
    x = torch.arange(start, end, step_size)
    y = torch.exp(-((x - (end / 2)) ** 2) / (1 / height))
    return y


# Example matrix with row vectors
matrix = torch.tensor([[1, 1, 1],
                       [4, 5, 6],
                       [1, 1, 1]])

# Define the interval and number of steps
start = 0
end = 1
num_steps = matrix.size()[0]  # num rows

# mapping weights vector
neg_log = neg_logarithmic_mapping(start, end, num_steps)
print('neg_log w: %s' % neg_log)
neg_lin = neg_linear_mapping(start, end, num_steps)
print('neg_lin w: %s' % neg_lin)
neg_gauss = neg_gaussian_mapping(start, end, num_steps)
print('neg_gauss w: %s' % neg_gauss)

weights = torch.tensor([0.2, 0.3, 0.5])
weighted = matrix * weights.view(-1, 1)
print('const w: %s' % weighted)

weights = neg_log
weighted = matrix * weights.view(-1, 1)
print('neg_log w: %s' % weighted)

weights = neg_lin
weighted = matrix * weights.view(-1, 1)
print('neg_lin w: %s' % weighted)

weights = neg_gauss
weighted = matrix * weights.view(-1, 1)
print('neg_gauss w: %s' % weighted)

# Calculate the weighted average
weighted_average = torch.mean(weighted, dim=0)

# Print the weighted average
print(weighted_average)

num_steps = 25
interval = end - start
step_size = interval / num_steps
x = torch.arange(start, end, step_size)
print(x.size()[0])
print(num_steps)
