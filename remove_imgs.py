import os
import random


fols=[i for i in range(10)]
for fol in fols:
    path='/home/sameenahmad/aryan1/fine_tune_dataset/sorted/'+str(fol)
    all_imgs=os.listdir(path)
    files_to_delete = random.sample(all_imgs, 2500)
    for file in files_to_delete:
        os.remove(path+'/'+file)



