
# coding: utf-8

#                                                        #Module-3
# # Notebook 1

# In[1]:


import sys
from math import sqrt

primeList = [2]
num = 3
isPrime = 1

while len(primeList) < 100000:
    sqrtNum = sqrt(num)

    # test by dividing with only prime numbers
    for primeNumber in primeList:

        # skip testing with prime numbers greater than square root of number
        if num % primeNumber == 0:
            isPrime = 0
            break
        if primeNumber > sqrtNum:
            break

    if isPrime == 1:
        primeList.append(num)
    else:
        isPrime = 1

    #skip even numbers
    num += 2

    
    
# print 1000th prime number
for n in range (9990,10000):
    print (primeList[n])


#saving output as text file
filename  = open("prime.txt",'w')
sys.stdout = filename
print("Prime Numbers from range :9991th to 10000th")


