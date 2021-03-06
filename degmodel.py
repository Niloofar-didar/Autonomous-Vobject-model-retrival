# -*- coding: utf-8 -*-
#  Affan + Nill (added some functions and changes) version

# this is to calculate regression model for degradation error , finding RMSE error and percentage error by changing interval parameters, gamma and max distance
'''
reads from all models in testvalue.csv and write information in degmodel.csv


modified regmodel for all objects where we can add  data for all objects to get their relative model

we can obseve that as gamma is smaller and closer to gamma1, the predicted model is closer to actual data in the closer distances and as gamma becomes begger, the predicted model is more closer to real data in farthesst distances. 
for gamma=gamma1 we can make distance confined to distance =20 as an eg ( or by  measuring virtual siize for object we can find a fixed ratio for distance that makes all object similar in terms of size and correspondingly calculate maximum confined distance for each one -> actually we have this feature allready as max distancef ro each object, we just need to change ratio to make max distance lower.
'''
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



exist=False

address='C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\degmodel_file.csv'
if(path.exists(address)):
    
  exist=True


#read in the file
#dataset = pd.read_csv("C:\\Users\\loaner\\Desktop\\newValues.csv")
dataset = pd.read_csv("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\one_deg.csv")
reader = csv.DictReader(open("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\one_deg.csv"))
tris_dataset = pd.read_csv("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\tris_fileseize.csv")

tris = tris_dataset['Tris'].values.reshape(-1,1) # points to the fillee with all objects
o_name = tris_dataset['name'].values.reshape(-1,1)
o_size = tris_dataset['size'].values.reshape(-1,1)


headerNames = []
file2 = open("Errors.txt","a+") 
file1 = open("SUM_Errors.txt","a+")
for i in reader.fieldnames:
    headerNames.append(i)

#length = len(headerNames)   #size of the original row headers

#epochs=length/6 # to the num of obj data
j=1

distance1 = (dataset['distance'].values.reshape(-1,1))
name1=dataset['name'].values.reshape(-1,1)
list11=(dataset['0.2'].values.reshape(-1,1))

objname=[]
dist=[]
list22=[]

for i in range(0, len(name1)):
    objname.append(str(name1[i][0]))
    list22.append(float(list11[i]))
    dist.append((distance1[i]))




print(objname[0] )
print( str(name1[0][0]))
print( float(distance1[0]))
count= distance1[len(distance1)-1] #num of objects
distance1=distance1[:len(distance1) -1]

mindis =[]
with open('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\degmodel_file.csv', mode='+a',newline='') as degmodel_file:
 
 counter2=0
 for ep in range(0, int(count)):

  name=objname[counter2+1]
  #if (objname[27] == "Cabin"):
  repetative=False

  
  #index= (temp_objname[::-1].index(name)) # last index of first objectobjname[::-1].index("Cabin")
  index=objname.index(name)
  while(index<len(objname) and objname[index]==name):
      index+=1
  
  
  
  multipleMultivariate = []

  first_regression = []
  second_regression = []
  third_regression = []
  final_regressionList = []
  newHeaders = []             #list to hold only the ratio values



  distance = dist[counter2:index]

  
  mindis.append(distance[0])
#gammaList = dataset['gamma'].values.reshape(-1,1)


#gamma = gammaList[0]

  counter = 1
  gamma_values = []
  filesize_values=[]
  
  #objname=dataset['name'+str(ep)].values.reshape(-1,1)
  list2=[]
  list1=[]
#Nill change
  list1=  list11[counter2:index]
  list2= list22[counter2:index]
  #dataset['0.2-'+str(ep)].values.reshape(-1,1)
  
  
  #row_count = sum(1 for row in reader)
  
  
  
  
  row_count = len(distance)
  gamma_1 = math.log ( list2[0]/list2[1], distance[1]/distance[0])
  gamma_2 = math.log ( list2[0]/list2[row_count-1], distance[row_count-1]/distance[0])

  gamma= (gamma_1 +gamma_2) /2
 

  gamma_values.append(gamma_1)
  gamma_values.append(gamma_2)
  gamma_values.append(gamma)

  
  file3=open("SUM_Errors.txt","r")
  data=[]
  data=file3.readlines()
  xx=0
  while(xx< len(data)):
      if (data[xx][:-1]==name):
          print ("data already exist")
          repetative=True
      xx+=4
  
    
  if(repetative==False):
    file2.write(str(objname[index-1])+"\n")
  
  
  
  print ("gamma1  is " + str(gamma_1) )
  print ("gamma2  is " + str(gamma_2) )
  print ("gamma_avg  is " + str(gamma) )


#("rowcount is "+str(row_count))

  ratio_list = []
  distance_list = []
  Y = []



  r=0
  for i in range( 0,4):
    ratio_list.append(round(r+0.20,2))
    r+=0.20



  for x in distance:
        distance_list.append(x)
        

  for i in ratio_list: # in ratios
    error_current = dataset[str(i)].values.reshape(-1,1)
    for x in range(0,row_count):
        Y.append(error_current[x])
        

  z1 = []
  z2 = []
  z3 = []

  tempZ3 = []

  for gamma_current in gamma_values:

    z1 = []
    z2 = []
    z3 = []
    
    tempZ3 = []
    
    for e in ratio_list:
        
        counter = 0
        for i in distance:
            value = (1 / ((distance[counter])**gamma_current))
            if counter < row_count:
                z3.append(value)
                tempZ3.append(value)
                counter = counter + 1
                
        counter = 0
        
        
    
        for i in tempZ3:
            value = float(e)*float(e)*i
            z1.append(value)
            counter = counter + 1
            
        counter = 0
        for i in tempZ3:
            value = i*float(e)
            z2.append(value)
            counter = counter + 1
            
        tempZ3.clear()
        
        X = [z1,
             z2,
             z3]
    

    
    Ya = np.array(Y)
    Xa=np.array(X).transpose()[0]
    reg = LinearRegression(fit_intercept=False).fit(Xa, Ya)
    
    
    
    
    
    
    coeffs=reg.coef_.tolist()
    coeffs.append(reg.intercept_)
    
    
    coeffs = [ round(elem, 4) for elem in coeffs[0] ]
    
    print('Multivariate Linear Regression Coefficients: ',coeffs)
    
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    
    multipleMultivariate.append(a)
    multipleMultivariate.append(b)
    multipleMultivariate.append(c)



  lowestError = []
  lowestErrorValue = 0
  determineGamma = []
  List_meanof_percerror=[]
  print()
  print()

  first_regression.append(multipleMultivariate[0])
  first_regression.append(multipleMultivariate[1])
  first_regression.append(multipleMultivariate[2])

  second_regression.append(multipleMultivariate[3])
  second_regression.append(multipleMultivariate[4])
  second_regression.append(multipleMultivariate[5])

  third_regression.append(multipleMultivariate[6])
  third_regression.append(multipleMultivariate[7])
  third_regression.append(multipleMultivariate[8])

  final_regressionList.append(first_regression)
  final_regressionList.append(second_regression)
  final_regressionList.append(third_regression)


  counter = 1
  newCounter = 0
  current_gamma = 0
  for current_coefficient in final_regressionList:
    
    a = current_coefficient[0]
    b = current_coefficient[1]
    c = current_coefficient[2]
    
    
    current_gamma = gamma_values[newCounter]
    lowestErrorValue = 0
    lowestError = []
    lowest_perc_err=[]
    print(counter,".", "CURRENT GAMMA BEING USED: ", current_gamma, "    current Regression being used: ", current_coefficient )
    print()
    for j in ratio_list: # fo all ratios we have
      model_values = []
      list2=[]
      new_distance=[]
      #j=j.split("-")[0]
      ratio = float(j)
    ############################################################
    #Nil 16 sep
    
      #for i in distance:
      interval=2
      #max_distance=distance[row_count-1]
      max_distance= 30
     # print ("interval is " + str( interval) + ", and MAx distance is " + str(max_distance)
    #)
      
      for i in range(0, row_count-4, interval):
    
        if ( distance[i] <=max_distance) : 
         #print ( str(distance[i]))
      #  print (str(ratio))
         denom = distance[i]**current_gamma
       # print ( str(denom))
        #Nil - correction in formula
         result = (((a*(ratio**2))+(b*ratio)+(c)) / (denom))
         model_values.append(result)
        else:
            break
      y_pred = model_values
    
      for i in range(0, row_count-4, interval):
        if ( distance[i] <=max_distance) : 
         list2.append( dataset[str(ratio)].values.reshape(-1,1) [i])
         new_distance.append( distance[i])
        else:
            break
    
        
      #list2=  dataset[str(ratio)].values.reshape(-1,1)
      #print ( str(list2))
    
      y_true = np.array(list2)
    
    #mse= mean_squared_error(list8, list6)
      mse= mean_squared_error(y_true, y_pred)
      rmse = np.sqrt(mse)
      lowestError.append(rmse)
      
      if(repetative==False):
        file2.write('RMSE (no intercept): {}'.format(rmse)+"\n")
        file2.write("percentage error for ratio "+ str(ratio)+ " is " + str(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)+"\n")
      #print('RMSE (no intercept): {}'.format(rmse))
    
      #print("percentage error for ratio "+ str(ratio)+ " is " + str(np.mean(np.abs((y_true - y_pred) / y_true)) * 100))
      lowest_perc_err.append(np.mean(np.abs((y_true - y_pred) / y_true) * 100))
    
        
    
    # nil 16 sep
    #############################################3
    
    
    lowestErrorValue = statistics.mean(lowestError)
    mean_perc_error= statistics.mean(lowest_perc_err)
    
    determineGamma.append(lowestErrorValue)
    List_meanof_percerror.append(mean_perc_error)
    
    
    denom = distance[i]**current_gamma
    result = (((a*(ratio**2))+(b*ratio)+(c)) / (denom))
    
    if(repetative==False):
      file2.write("Average RMSE error: "+ str( lowestErrorValue)+"\n")
    
    print("Average RMSE error: "+ str( lowestErrorValue))
    #print ( str(y_pred))
    
    
    plt.rc('font', size=5) 
    plt.xticks(distance)
    plt.plot(new_distance, model_values) # both should have same dimension
    print()
    counter = counter + 1
    newCounter = newCounter + 1


  indx = 0
  temp = determineGamma[0]
  for i in (0,len(determineGamma)-1):
    if temp > determineGamma[i]:
        temp = determineGamma[i]
        indx = i

  indx = indx % 3
  #print("Lowest AvG RMSE out of all the Gamma values: ", temp, " for Gamma value:", gamma_values[indx])
  
  if(repetative==False):
    file1.write(str(objname[index-1])+"\n")
    file2.write("Lowest AvG RMSE out of all the Gamma values: "+ str(temp)+ " for Gamma value:"+str( gamma_values[indx])+"\n")
    file1.write("Lowest AvG RMSE out of all the Gamma values: "+ str(temp)+ " for Gamma value:"+str( gamma_values[indx])+"\n")
  

  indx = 0
  temp = List_meanof_percerror[0]
  for i in (0,len(List_meanof_percerror)-1):
    if temp > List_meanof_percerror[i]:
        temp = List_meanof_percerror[i]
        indx = i

  indx = indx % 3
  print("Lowest AvG Percentage_err out of all the Gamma values: ", temp, " for Gamma value:", gamma_values[indx])
  if(repetative==False):
    file2.write("Lowest AvG Percentage_err out of all the Gamma values: "+ str(temp)+ " for Gamma value:"+ str(gamma_values[indx])+"\n")
    file1.write("Lowest AvG Percentage_err out of all the Gamma values: "+ str(temp)+ " for Gamma value:"+ str(gamma_values[indx])+"\n")
    file1.write("\n")
    file2.write("\n")
  z1 = []
  z2 = []
  z3 = []

  for e in ratio_list:
    #e=e.split("-")[0]
    counter = 0
    for i in distance:
        value = (1 / ((distance[counter])**gamma_values[indx]))
        if counter < row_count:
            z3.append(value)
            tempZ3.append(value)
            counter = counter + 1
            
    counter = 0
    
    

    for i in tempZ3:
        value = float(e)*float(e)*i
        z1.append(value)
        counter = counter + 1
        
    counter = 0
    for i in tempZ3:
        value = i*float(e)
        z2.append(value)
        counter = counter + 1
        
    tempZ3.clear()
    
    X = [z1,
         z2,
         z3]




  Ya = np.array(Y)
  Xa=np.array(X).transpose()[0]
  reg = LinearRegression(fit_intercept=False).fit(Xa, Ya)






  coeffs=reg.coef_.tolist()
  coeffs.append(reg.intercept_)


  coeffs = [ round(elem, 4) for elem in coeffs[0] ]
  print("data for obj "+ str(ep) + ": ")

  print("based on RMSE values,")
  print('Most Accurate Multivariate Linear Regression Coefficients: ',coeffs)
  #j=k+1
  #k+=6
  
  o_name2 = o_name.tolist()
  obj_indx= o_name2.index(name1[index-1]) # search deg model obj name in lis of objects to find the index for tris
  
  if(repetative==False):
    file_writer = csv.writer(degmodel_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    if(exist==False):
      file_writer.writerow(['alpha','betta','c','gamma','max_deg','tris','name','mindis', 'filesize'])
 
    file_writer.writerow([coeffs[0], coeffs[1], coeffs[2],gamma_values[indx],float(list1[0]), float(tris[obj_indx]),str(objname[index-1]), float(mindis[ep]), float(o_size[obj_indx]) ])
  
  counter2=index
  
file2.close()
file1.close()
file3.close()