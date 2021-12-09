import sys
import random

f = open(sys.argv[1]+'.txt', 'r')
op1 = f.readlines()
f.close()
op1 = [i.rstrip().split() for i in op1]
op1_name = sys.argv[1]

f = open(sys.argv[2]+'.txt', 'r')
op2 = f.readlines()
f.close()
op2 = [i.rstrip().split() for i in op2]
op2_name = sys.argv[2]

M = 10000
B = 10
eps = 0.1
op1_N = 0
op2_N = 0


for m in range(M):
    op1_n = 0
    op2_n = 0
    for b in range(B):
        idx = random.randint(0, 124)
        if float(op1[idx][-1]) > float(op2[idx][-1]):
            op1_n += 1
        else: op2_n += 1
    if op1_n > op2_n:
        op1_N += 1
    else:
        op2_N += 1

print(op1_name, op1_N/M)
print(op2_name, op2_N/M)

