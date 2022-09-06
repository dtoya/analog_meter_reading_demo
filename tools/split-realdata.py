
import os
import re
import random

split = 0.8
root_dir = './data/realmeter'
image_dir = root_dir + '/images'
file_list = os.listdir(image_dir)
total_size = len(file_list)
train_size = int(total_size * split)
val_size = total_size - train_size
print('total_size = {} train_size = {} val_size = {}'.format(total_size, train_size, val_size))
#input()

index = list(range(total_size))
random.shuffle(index) 
index_train = index[:train_size]
index_val = index[train_size:]

if (len(index_train) != train_size) or (len(index_val) != val_size):
    print("Error: len(index_train) = {} train_size = {} len(index_val) = {} val_size = {}".format(len(index_train), train_size, len(index_val), val_size))
    exit()

with open(root_dir+'/train.txt', 'w') as f:
    for i in index_train:
        f.write('%s\n'%(file_list[i]))

with open(root_dir+'/val.txt', 'w') as f:
    for i in index_val:
        f.write('%s\n'%(file_list[i]))
    
