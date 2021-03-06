# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:12:31 2021
generates deg model of csv file distance and ratios -> feed into degmodel.py to generate reall coefficients and then write error results and model into degmodel.csv
this code reads from output of python for deg model, reads distanes.tct and  Degradation_Error.txt' and writes into onedeg.txt

output here is onedeg.csv but it generates as many as models you need

@author: gx7594
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import csv
import statistics
import math 
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import os.path
from os import path




address="C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\one_deg.csv"
already_objects=0
exist=False
name2=[]
if(path.exists(address)):
    
  exist=True
  dataset = pd.read_csv("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\one_deg.csv")
  name1=dataset['name'].values.reshape(-1,1)

  x=0
  while(x<len(name1)):
      name2.append(str(name1[x][0]))
      x+=1
  #name2=list(name1)
  if(len(name2) !=0):
   already_distance=dataset['distance'].values.reshape(-1,1)
   already_objects=already_distance[len(already_distance)-1]
   #dataset['distance'].values.reshape(-1,1)[len(already_distance)-1]=None
   dataset=dataset.iloc[0:len(already_distance)-1,0:6]

   dataset.to_csv(r'C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\one_deg.csv'
, index = False, header=True)


with open('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\one_deg.csv', mode='+a',newline='') as degmodel_file, open('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\Degradation_Error.txt', mode='r') as deger ,  open('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\Distances.txt', mode='r') as dist:
 
    
       file_writer = csv.writer(degmodel_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      
       if(exist ==False):
          file_writer.writerow(['distance','0.8','0.6','0.4','0.2','name'])
      
    
        #
        
       errors = deger.readlines()
       distances1=dist.readlines()
       count=int(distances1[0])
       distances1=distances1[1:]
       iterator=0
       written_data=0
       while(iterator<count):
       
         
          j=0  
          j= int(distances1.index('\n')  )
        
          distances= distances1[:j]
          i=0

          name=str(distances[j-1][:-1])
          
          #if(exist==True and len(name2)!=0):
          
          if( name2.count(name)<=0):  
            written_data+=1
            eror_lines=(len(distances)-1) # -2 since the end of distance is name of object  
            #j=eror_lines

            er = [[]]*4
            for k in range(1,5 ): # four decimations
        
              er[k-1]=errors[eror_lines*(k-1) +k-1 :(eror_lines*k) +k-1   ]
            
            
            last_index_error=   (eror_lines*k) +k-1
    
            #er[0][1]=er[0][1][:-1]   removes \n in each cell
            #print (er[0][1])
        
            #print er[[]]
            #print er[2][i]
            while i< len(distances)-1:
               print (er[0][i])
               file_writer.writerow([ distances[i][:-1],str(er[0][i][:-1]),er[1][i][:-1],er[2][i][:-1],er[3][i][:-1],str(distances[j-1][:-1]) ])
               i+=1
               
 
            
            if(iterator<count):
               distances1=distances1[j+1: len(distances1)]
               errors=errors[last_index_error+1: len(errors)]
           
          iterator+=1 
       file_writer.writerow([written_data+ int(already_objects)])
       
