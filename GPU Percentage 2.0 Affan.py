# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 06:35:45 2020

@author: Affan Atif -> NIl changed totally
tries to find pivot for the first zone of gpu model
"""

from itertools import *
import pandas as pd  
import numpy as np  
import csv
import array as arr 
import math 
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
import statistics
dataset = pd.read_csv("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\OCT\\GPU.csv")



dataset.plot(x='Tris1', y='GPU1', style='o')  
plt.title('GPU vs Tris')  
plt.xlabel('Tris')  
plt.ylabel('GPU Percentage')  
plt.show()

sizeArray = 0

#X = dataset['Tris1'].values.reshape(-1,1)
#y = dataset['GPU1'].values.reshape(-1,1)


def cal_mean(array):
    sum=0;
    for i in array:
         sum+=i
         
    mean= sum/len(array)  
    return mean





with open("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\OCT\\GPU.csv") as f:
    reader = csv.reader(f)
    i = (next(reader))
  
    
total_tris=[]  
total_mingpu=[]  
column=len(i)    
print(column)



for count in range(1, int(column/5)+1, 1):
#for i in range(1, 2, 1):
    
    x1=[]
    y1=[]
    X=[]
    y=[]
    y2=[]
    x2=[]
    #if(count==30):
      #count+=1
      
    if(count!=1  and count!=2 and count!=4
       and count!=11 and  count!=16 and count!=19 and count!=22 and count!=23 
        and count <27):
       #!=27 and count!=23 and count!=30 and count!=32   )   :
      
     x1= ( dataset['Tris'+str(count)].values.reshape(-1,1))
     y1=(dataset['GPU'+str(count)].values.reshape(-1,1))
     objname=(dataset['name'+str(count)].values.reshape(-1,1))

     dataset.plot(x=('Tris'+str(count)), y=('GPU'+str(count)), style='o')  
     plt.title(str(objname[0]))  
     plt.xlabel('Tris')  
     plt.ylabel('GPU Percentage')
     plt.show()
    
     j=1
     while(j<=len(y1)):
    
        if((pd.isnull(x1[j-1]))):
            break
        else:
            
            
            x2.append(float(x1[j-1]))
            y2.append(float(y1[j-1]))
            j=j+1
           # y2.append(float(y1[j-1]))


    
     y= np.array(y2).reshape(-1,1)
     X= np.array(x2).reshape(-1,1)
    #print (y)
     print ("obj "+ str(count))
     gp=3
     total_points=len(y)
     subg_c= (int) (total_points/3) #having 3 elms in each gp
     subg=[[]] *subg_c
   
     start=0
     mean_gpu_subg=[]
     for i in range( 0, subg_c ): # gets all points except the last one
        start =  (int) (i * (total_points/subg_c))
        end = (int) (start + (total_points/subg_c))
        subg[i] = y[ start : end ]
        mean_gpu_subg.append( 0)
        mean_gpu_subg[i]= cal_mean(subg[i])
        
   
     min_ofmean= min(mean_gpu_subg)
     min_of_meandex= mean_gpu_subg.index(min_ofmean) # index of minimum of means in subgroups-> best index of subgp
     min_of_subg=( min(subg[min_of_meandex] )  )   
     tmpindex =y2[gp*(min_of_meandex):].index(min_of_subg)
   
     mindex= (gp*min_of_meandex) +tmpindex
     #mindex = y2[gp*(min_of_meandex):gp*(1+min_of_meandex)].index(min_of_subg)
     print (mindex,float(min_of_subg) )
    
     '''
     for i in range (mindex, len(y)-1):
        if y[i+1] - float( min_of_subg) <2:
            mindex= i+1
            print (mindex)
     '''
     total_tris.append(X[mindex ])
     total_mingpu.append(min_of_subg)
    
    
    
     print(" tris "+ str(X[mindex ]) +" gpu "+ str(y[mindex]))
    
total_mingpu1=[]

for el in total_mingpu:
   total_mingpu1.append(el)

for k in  total_mingpu1:
   if (  float(k) >48 ):
            index= total_mingpu.index(k)
            total_tris.remove(  total_tris[index])
            total_mingpu.remove( k )
           
            
            
print ((total_tris))
print (" min "+ str(min(total_tris)) + " max "+ str (max (total_tris))+ " mean "+ str (cal_mean(total_tris)))

print("mean betwee without plane and with plane scenarios = " + str((48101 + 36711)/2))
   
'''
    sizeArray = len(X)
    #print(sizeArray)

    newArray = []
    minValues = []
    tempArray = []
    percentage = []

    i = 1
    x = 0

    newArray = []
    i = 0
    while(i < sizeArray):
      newArray.append(y[i])
      minValues.append(y[i])
      i = i + 1
    


    i = 0
    temp = 0
    minValues[0] = 1000000
    minValues[1] = 1000000
    minValues[2] = 1000000
    minValues[3] = 1000000
    minValues[4] = 1000000
    minValues[5] = 1000000
    minValues[6] = 1000000

#print("lowest", min(minValues))
    tempFinal = min(minValues)

    newCounter = 0
    finalIndex = 0
    testArray = []
    threshold = 0
    for i in newArray:
     if (i==min(minValues)):
        print("Index of Lowest", i, min(minValues), newCounter)
        finalIndex = newCounter+2
     newCounter = newCounter + 1
    
    
    newCounter = 0
    for i in y:
     if (i==97):
        threshold = newCounter
        newCounter = newCounter + 1
        break
     newCounter = newCounter + 1
     
    threshold = threshold + 2
    newArray.reverse()



    newCounter = 0
    endZoneIndex = 0
    for i in newArray:
     if (i==max(newArray)):
        print("max counter", newCounter+2, i)
        endZoneIndex = newCounter+2
     newCounter = newCounter + 1


#final index value for 2nd zone has now been calculated

    endZoneIndex = len(newArray) - endZoneIndex + 3
    X_new = []
    Y_new = []

    print("Initial index", finalIndex)
    print("Final index", endZoneIndex)
    
    
    remove it
    '''
    
    
    
'''
    for i in range(finalIndex, endZoneIndex-1):
   
     X_new.append(X[i-2])
     Y_new.append(y[i-2])



    X_train, X_test, y_train, y_test = train_test_split(X_new,Y_new, test_size=0.2, random_state=0)


    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)


    print("Intercept of line:", regressor.intercept_)
    #For retrieving the slope:
    print("Slope of Line:", regressor.coef_)


    y_pred = regressor.predict(X_test)


    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("Index for 97%:", threshold)
    '''




