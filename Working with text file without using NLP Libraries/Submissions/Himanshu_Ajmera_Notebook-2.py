
# coding: utf-8

#                                                     Module 3
# # Notebook-2

# a. Create an “encode_rot()” function to encode any given strings using ROT algorithm. The input should contain a key and a string of text. The key can be any integers both negative and positive (-12: turn left 12 positions, 36: turn right 36 positions). Only alphabet letters are encoded.
# 

# In[3]:


import string
def encoder_rot(s,n):
    p = n%26
    return s.translate(
        str.maketrans(
            string.ascii_uppercase + string.ascii_lowercase,
            string.ascii_uppercase[p:] + string.ascii_uppercase[:p] +
            string.ascii_lowercase[p:] + string.ascii_lowercase[:p]
        )
    )


# In[4]:


#testing the function   
clear_text="Machine CAN learn 2!!!"
print(encoder_rot(clear_text,28))


# b. Create a decode_rot() to decode a ciphertext. The input only contains the ciphertext. The output contains the cleartext and the key that was used to encode text. The key will be between 0 and 25. (hint: Compare your decoded clear text with a dictionary text file and decide which one has the most dictionary words.)

# In[11]:


import sys
import os
import re

def decode_rot(cipher_text):
    # define ekey as temp key 
    temp_key = 1
    max_Count = 0
    plain_text = ''
    # alphabet range 1-26
    for key in range(1,26):
        check_text = ''
        
        for c in cipher_text:
            order = ord(c)
            if  not c.isupper() and not c.islower() or c.isdigit():
                check_text += chr(order)
            else:
                # If char stream and rotation is in the upper case
                if c.isupper() and chr(order - key % 26).isupper():
                    shift = order - key % 26
                    check_text += chr(shift)
                # If char stream and rotation is in the lower case 
                elif c.islower() and chr(order - key % 26).islower():
                    shift = order - key % 26
                    check_text += chr(shift)
                # Otherwise move back 26 spaces after rotation.
                else: # alphabet range 1-26
                    shift = order - (key % 26 - 26)
                    check_text += chr(shift)
        
        
        p = re.compile(r'\W+')
        words = p.split(check_text)
        count = 0
        
        # read dictionary file and compare it line by line with the text words
        for word in words:
            word = word.lower()
            for line in open('dictionary.txt','r').readlines():
                dictword = line.split()
                if word in dictword:
                    count += 1
                    break
        
        # calculating the exact key from the list 1-26 
        if count > max_Count:
            max_Count = count
            temp_key = key
            plain_text = test_text
    
    return plain_text, temp_key




# In[13]:


#testing thee function
cipher_text = 'Tqjq yi byau fuefbu, ydjuhhewqju yj xqht udekwx qdt yj mybb jubb oek mxqjuluh oek mqdj je xuqh.'
plain_text, key = decode_rot(cipher_text)
print('\nThe plain text is :', plain_text)
print('The key is', key)

