import torch
import random
import math

num = 0
singular_list = torch.zeros(100)

while num < 100:
    x = 2*random.random()
    y = random.random()
    res = math.sqrt(4-x*x)/math.pi
    if y <= res:
        singular_list[num]=x
    num +=1

print(singular_list)
