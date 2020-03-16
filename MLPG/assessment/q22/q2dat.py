import matplotlib.pyplot as plt
import itertools
L = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
S = 181,198,96,119,69,108,91,86,82,65,194,97,69,68,139,137,71,131,145,55,108,165,110,76,156,92,131,146,193,142,123,107,93,149,93,139,145,190,195,148,118,182,116,136,79,96,198,147,175,92,71,173,141,205,109,91,85,76,209,220,194,245,204,211,145,227,86,146,130,196,172,192,171,205,224,150,223,99,104,186,119,240,222,231,106,227,130,231,100,78
A = 33,11,41,92,66,89,70,98,97,58,41,20,32,80,83,99,24,82,90,78,14,86,92,23,36,74,70,19,88,99,77,19,30,94,58,83,18,98,3,59,79,23,56,84,7,48,62,34,69,54,62,96,119,91,113,24,115,32,86,71,144,113,54,31,107,22,60,84,145,129,92,14,130,104,139,43,48,55,52,43,36,114,31,30,130,13,23,21,126,51
P = 100,106,54,59,41,53,52,44,39,31,100,58,42,39,73,69,44,71,73,32,57,84,55,41,83,52,69,86,96,74,65,58,50,73,49,66,79,101,103,74,61,96,59,69,49,54,103,81,87,51,78,169,138,195,121,86,99,75,178,187,190,223,185,178,145,187,82,141,137,192,163,161,171,198,216,140,201,100,101,164,104,222,181,200,109,188,114,187,110,80

zero_loc = L.index(1) - 1
one_loc = L.index(1)
print(zero_loc)
L0 = L[0:zero_loc]
S0 = S[0:zero_loc]
A0 = A[0:zero_loc]
P0 = P[0:zero_loc]

datadkt = {
    'L': L,
    'S': S,
    'A': A,
    'P': P,
    'N': 90 #number of datapoints, first 50 are L=0, last 40 are L=1
}

new_S, new_P = zip(*sorted(zip(S0, P0)))
plt.plot(new_S,new_P)
plt.xlabel("Size")
plt.ylabel('Price')
plt.show()

new_A, new_P = zip(*sorted(zip(A0, P0)))
plt.plot(new_A,new_P)
plt.xlabel("Age")
plt.ylabel('Price')
plt.show()

new_L, new_P = zip(*sorted(zip(L0, P0)))
plt.plot(new_L,new_P)
plt.xlabel("Locality")
plt.ylabel('Price')
plt.show()