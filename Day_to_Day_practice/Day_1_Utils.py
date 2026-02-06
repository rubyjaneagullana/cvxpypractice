import numpy as np

f_c = 4e6
f_s = 1e6
N = 1024

T = 1 / f_s
t = np.arange(N) * T

def generate_signal(f_c, t):
    return np.cos(2 * np.pi * f_c * t)


print(generate_signal(f_c,t))


#Printing db2li

def db2lin(db):
    return 10 **(db/10)

db = 100
print(db2lin(db))