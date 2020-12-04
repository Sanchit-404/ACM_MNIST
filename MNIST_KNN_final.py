
# In[7]:

import operator
import csv
import numpy as np
import os
os.chdir("E:\\ACM_task\\KNN\\raw_files")


# In[2]:



def CalcDistance(input_1, input_2, length):
    distance = 0
    for i in range(length-1):
        distance +=(input_1[i] - input_2[i])**2
    Euclidean_distance = distance**(1/2)
    return Euclidean_distance 


# In[11]:



def neighbours(train_matrix_final, testInstance, k):
    Neighbours_distances = []
    for i in range(len(train_matrix_final)):
        respective_distance = CalcDistance(testInstance, train_matrix_final[i,1:785],len(testInstance))
        
        Neighbours_distances.append((train_matrix_final[i],respective_distance))
   
    Neighbours_distances.sort(key=operator.itemgetter(1))
   
    Final_neighbors = []
    for i in range(k):
        Final_neighbors.append(Neighbours_distances[i][0])
    return Final_neighbors


# In[12]:


def best_neighbours(find_neighbours):
    neighbour_count = {}
    for x in range(len(find_neighbours)):
        occurrence =find_neighbours[x][0]
        if occurrence in neighbour_count:
            neighbour_count[occurrence] += 1
        else:
            neighbour_count[occurrence] = 1 
    BestNeighbour = sorted(neighbour_count.items(), key=operator.itemgetter(1), reverse=True)
    return BestNeighbour[0][0]


# In[10]:


def main():
    
    with open('mnist_train.csv', newline='') as csv_file1:
    
        train_data_lines = csv.reader(csv_file1)
        train_dataset=list(train_data_lines) 
        
        train_matrix=np.array(train_dataset).astype("int")
    
    with open('mnist_test.csv', newline='') as csv_file2:
    
        test_data_lines = csv.reader(csv_file2)
        test_dataset=list(test_data_lines)
        
        test_matrix=np.array(test_dataset).astype("int")

    predictions=[]     
    k=1
    for i in range(len(test_dataset)):
        find_neighbours=neighbours(train_matrix,test_matrix[i],k)
        
        result = best_neighbours(find_neighbours)
        
        predictions.append(result)
        print('Real Number is:' + repr(test_matrix[i,0])+' Predicted Number :' + repr(result))
    
              
main()


# In[ ]:




