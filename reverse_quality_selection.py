# -*- coding: utf-8 -*-
"""

NO NEEED TO THIS CODE ANYMORE----> WE HAVE EQUAVALENT OF TEST.PY IN UBUNTU THAT NOT ONLY FINDS THE QUALITIES
BUT ALSO IT GENERATES SCREENSHOTS


Created on Thu Feb 25 12:23:25 2021

@author: gx7594

equal code is test.py in libmsd directory ubuntu

"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 22:14:26 2020
optimization algorithm and
the function to select which quality is needed having factor*maxdecimation for each object
then calculating the relative quality at selective distances to generate them in blender

@author: gx7594
"""
import random
import statistics
import math 
from math import sqrt
import pandas as pd  
import numpy as np  
import array
import csv
import array as arr 
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score



def QualitySelection_reverse(ind,d11): # this is to find candidate qualities based on factor * max_deg 
# factors are 0.2 . 0.4 and 0.6
     
    i=ind
    gamma=float(gamm[i])
    a= float(alpha[i])
    b=float(beta[i])
    c1=float(c[i])
    
    
  
    c3=c1- ((d11** gamma)* max_deg[i] * 0.6)
    c2= c1- ((d11** gamma)* max_deg[i] * 0.4)
    c1=c1-((d11** gamma)* max_deg[i] * 0.2) # ax2+bx+c= (d^gamma) * max_deg


    inp1=[1.0,1.0,1.0]
    inp2=[1.0,1.0,1.0]
 
    #print("first equation")
    r1,r2=delta(a,b,c1,c[i],d11,gamma,max_deg[i] * 0.2 )
    r1= float(round(r1,2))
    r2=float(round(r2,2))
    
    #inp1.append(r1)
   # inp2.append(r2)
    inp1[0]=r1
    inp2[0]=r2
    #print("second equation")
    
    r1,r2= delta(a,b,c2,c[i],d11,gamma, max_deg[i] * 0.4)
    r1= float(round(r1,2))
    r2=float(round(r2,2))
    inp1[1]=r1
    inp2[1]=r2

    r1,r2= delta(a,b,c3,c[i],d11,gamma, max_deg[i] * 0.6)
    r1= float(round(r1,2))
    r2=float(round(r2,2))
    inp1[2]=r1
    inp2[2]=r2
   
    
    
    
    #need fixing !!!
    sel_inp=0 # selects amon inp1 or inp2
    final_inp = array.array('f')
    final_inp=[1.0,1.0,1.0]
  
   
    for i in range (0,3):
        
        if(inp1[i]==0 and inp2[i]==0):
            final_inp[i]=1.0
        
        elif( inp2[i]==0 or inp2[i]==1)  :
            final_inp[i]=inp1[i]
        else:
            if( inp1[i]==0 or inp1[i]==1): 
              final_inp[i]=inp2[i]
            else:
                 final_inp[i]=min (inp2[i], inp1[i])
         
    return final_inp
     
     
def delta(a,b,c,c_real,d,gm, max_d):
  #("Quadratic function : (a * x^2) + b*x + c=0")
  r=0.0
  r = (b**2 - 4*a*c)

  if r > 0:
    num_roots = 2
    x1 = (((-b) + sqrt(r))/(2*a))     
    x2 = (((-b) - sqrt(r))/(2*a))
    #print("There are 2 roots: %f and %f" % (x1, x2))
    if(0.1<x1<1.0 and 0.1<x2<1.0):
      return x1,x2
    if (0.1<x1<1) :
      x=checkerror(a, b, c_real, d, gm, max_d) 
      #x=1
      return x1,x
    if (0.1<x2<1) :
      x=checkerror(a, b, c_real, d, gm, max_d)
      #x=1
      return x,x2
    else:
      x=checkerror(a, b, c_real, d, gm, max_d)
      #x=1
      return x,x
  
    
  elif r == 0:
    num_roots = 1
    x = (-b) / 2*a
    #print("There is one root: ", x)
    return x,0
  else:
    num_roots = 0
    #print("No roots, discriminant < 0.")
    x=checkerror(a, b, c_real, d, gm, max_d)
    #x=1
    return x,x
    #0 is delta1 here      
     


def checkerror(a,b,creal,d,gamma, max_d):
    
    
    r1=0.1
    error=max_d
    for i in range (1,18):
        
     error = ((a* (r1**2)) + b*r1 + creal) / (d**gamma)
     if(error<max_d):
       return r1
     r1+=0.05
     #r2=0.8
     #error2 = ((a* (r2**2)) + b*r2 + creal) / (d**gamma)
    '''
    if(error1<max_d):
     return r1
    elif(error2<max_d):
     return r2
    else:
     
     # 0 is as delta1 or r2=1
     '''
    return 0
     
    
     
    
with open("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\degmodel_file.csv", newline='') as csvfile:
 dataset = csv.reader(csvfile, delimiter=' ', quotechar='|')
    
#dataset = pd.read_csv("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\degmodel_file.csv")
for row in dataset:
       print(', '.join(row))

upos_x=0
upos_y=0
#uspeed=1 #1m/s user speed
'''uspeed x and uspeedy could be either +, - , or 0 ;; if positive it means user goes from left to right
if negative, user moves fr right to left, it helps in updating user position for prediction or after each decision epoch'''
uspeedx=0.5
uspeedy=0
uspeed = math.sqrt( (uspeedx)**2 + (uspeedy)**2 )
#desicion_p=3 #each 3 sec we decide

'''gpu model= a tris + b :  4.10365E-05 tris + 44.82908722

agpu=4.10365E-05
bgpu=44.82908722

decision_epoch=[]

'''

obj_x=[]
obj_y=[]
#min_dis=[] # minimum distance between user and each obj -> let it be similar for all objects
us_ob_dis=[]
totaltris=0
eng_blc=[]
eng_dec=[]
gpusaving=[]
eng_transf=[]
t_transf=[]
test=[]
t_dec=[] # stores worse case decimation which is from 1 to 0.9 as eg,, from 1 to 0.2 is 0.2 * (t of 0.9)
#max_deg_error=0.01 # should be a list for all objs indeed/ this is for simplification 
#gpu = 3e-5 t + 45

tris = dataset['tris'].values.reshape(-1,1) # points to the fillee with all objects
o_name = dataset['name'].values.reshape(-1,1)

## for degradation error formula deg= ax2 + bx + c / (d**y)
alpha= dataset['alpha'].values.reshape(-1,1)
beta= dataset['betta'].values.reshape(-1,1)
c= dataset['c'].values.reshape(-1,1)
gamm= dataset['gamma'].values.reshape(-1,1)
max_deg= dataset['max_deg'].values.reshape(-1,1)
mindis=dataset['mindis'].values.reshape(-1,1)
i=0

obj_count=len(alpha)

while i<len(alpha):
 test.append(-beta[i]/(2*alpha[i]))
 i+=1
 
 


visible=[] # check obj in FOV = y variable
closer=[] # check if user is gettiing closer or farther = z variable
decimate=[]
decimatable=[]
prev_quality=[]
best_cur_eb=[]
quality_log=[]
distancelog=[]
closerlog=[]
predictlog=[]
obj_backward=[]
logbestpredicteb=[]
cache=[]
fthr=[]
distance = [2,4,6]

objects_quality=[] 
with open('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\qualities_reversal.txt', mode='w',newline='') as qual:
    
 for i in range(0, obj_count):
  #obj_x.append(random.randint(0, 5))
  #obj_y.append(random.randint(0, 5))
  name=str(o_name[i])
  
  qual.write(str(name[4:-4])+ "\n")
  obj_backward.append(False)
  #min_dis.append(1)
  #us_ob_dis.append(Distance(upos_x, obj_x[i], upos_y, obj_y[i]))
   
  decision_epoch.append(1)   
  totaltris+=tris[i]  
  
  closer.append(True)
  visible.append(0) #y 
  decimate.append(0)# if we'll send decimation req
  decimatable.append(0) # if decimate is possible in period p while p is fixed 
  '''assume we decide about next period in the starting point of current period'''
  eng_blc.append(0)
  eng_dec.append(0)                 
  gpusaving.append(0)   
  eng_transf.append(1)   
  t_transf.append(0.4) 
  t_dec.append(1)
  prev_quality.append(1)
  best_cur_eb.append(0)
  quality_log.append("")
  distancelog.append("")
  closerlog.append("")
  predictlog.append("")
  logbestpredicteb.append("")
  cache.append(1)
  fthr.append(1)
  
  #
  temp=[]
  j=0
  for elm in distance:
   temp.append(QualitySelection_reverse(i, elm))
   temp3=(temp[j])
   #temp3=temp2.split(",")
   
   qual.write(str(elm)+" " + str(temp3[0])+" "+ str(temp3[1]) +" "+str(temp3[2])+" " +"\n")
   j+=1
  
  objects_quality.append(temp)
  
    
      
  


