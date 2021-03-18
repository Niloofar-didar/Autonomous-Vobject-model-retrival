# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:31:21 2021
GENERATE SCREENSHOTS SIDE BY SIDE +

code to generate gpu graph and data base graphs 

@author: gx7594
"""

""
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
from matplotlib import pyplot as plt2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
'''
START sECTION FOR SURVEY GRAPHS 
READS FROM MODEL.TXT WHICH IS OUTPUT FOR SCREEN SHOTS OF TEST.PY IN LIBMSD DIRECTORY

'''
'''
img = mpimg.imread('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Jan21\\planefact0.2d4.0r0.65.bmp')

img2 = mpimg.imread('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Jan21\\planefact0.2d2.0r1.0.bmp')

plot_image = np.concatenate((img, img2), axis=0) # horizantally axis =1
fig = plt.figure(figsize=(50, 50))
plt.imshow(plot_image)

plt.savefig('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Jan21\\PLANE.png')

#imgplot = plt.imshow(img)
plt.show()
'''


'''
END sECTION FOR SURVEY GRAPHS 
'''



'''
START sECTION FOR DATA BASE GRAPH
'''
markers = ['>', '+', '.', 'o', 'v', 'x', 'X', '|','^','<','s','p','*','h','H','1']


plt.figure()
# Hold activation for multiple lines on same graph
#plt.hold('on')
# Set x-axis range
plt.xlim((1,9))
# Set y-axis range
plt.ylim((1,9))
# Draw lines to split quadrants
plt.plot([5,5],[1,9], linewidth=3, color='blue' )
plt.plot([1,9],[5,5], linewidth=3, color='blue' )
plt.title('Quadrant plot')
# Draw some sub-regions in upper left quadrant
#plt.plot([3,3],[5,9], linewidth=2, color='blue')
#plt.plot([1,5],[7,7], linewidth=2, color='blue')
plt.show()


#############
'''
sECTION FOR DATA BASE GRAPH
'''
hightris_highvol=[]
hightris_lowvol=[]
lowtris_highvol=[]
lowtris_lowvol=[]
obj=0
with open('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Jan21\\size_data.txt', mode='r') as input:


 inp = input.readlines()
 volume=[]
 tris=[]
 area=[]
 texture=[]
 temp=[]
 name=[]
 
 i=1
 for elm in inp:
     temp= elm.split(" ")
     volume.append(float(temp[1])) #volume
     area.append(float(temp[2])) #area
     tris.append(int(temp[3])) #
     name.append(temp[0])
     temp2= temp[4]
     texture.append(str(temp2[:-1]))
     i+=1
 
    
 max_vol= max(volume)
 max_tris= max(tris)
 
 log_vol=math.log(max_vol,2)
 log_tris=math.log(max_tris,2)
 
 for i in range (0, len (volume)) : # to the number of objects
  #if(texture[i]=="No"):
      
   #plt.scatter(volume[i], tris[i], c='r',marker='x')
   #plt.scatter(area[i], tris[i], c='b',marker='o') 
  #else:
   #plt.scatter(volume[i], tris[i], c='b',marker='o')
   #plt.scatter(area[i], tris[i], c='b',marker='o') 
   
  '''
   
  if (volume[i] <= 0.00779 and tris[i] <= 35000):
       lowtris_lowvol.append(name[i])
       
  elif(volume[i] <= 0.00779 and tris[i] > 35000)  :    
       hightris_lowvol.append(name[i])
 
  elif(volume[i] > 0.00779 and tris[i] > 35000)  :    
       hightris_highvol.append(name[i])
 
  elif(volume[i] > 0.00779 and tris[i] <= 35000)  :    
        lowtris_highvol.append(name[i]) 
  '''
 
  if (area[i] <= 0.6 and tris[i] <= 35000):
       lowtris_lowvol.append(name[i])
       plt.scatter(area[i], tris[i],marker='o', color='r') 
       
  elif(area[i] <= 0.6 and tris[i] > 35000)  :    
       hightris_lowvol.append(name[i])
       plt.scatter(area[i], tris[i],marker='X', color='b') 
 
  elif(area[i] > 0.6 and tris[i] > 35000)  :    
       hightris_highvol.append(name[i])
       plt.scatter(area[i], tris[i],marker='h', color='orange') 
 
  elif(area[i] > 0.6 and tris[i] <= 35000)  :    
        lowtris_highvol.append(name[i])
        plt.scatter(area[i], tris[i],marker='v', color='m') 
        
        
 plt.legend([ "HTris_LArea", "HTris_HArea","LTris_HArea", "LowTris_LowArea"  ],
           loc="upper center", bbox_to_anchor=(1,1 ), fontsize=11)
            #loc="upper center", bbox_to_anchor=(0.8,-0.2 ), fontsize=11)
 #plt.legend(loc=[1.1, 0.5])  

 #axx = plt.subplot(111)
 #box = axx.get_position()
 #axx.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])       
 #axx.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),   fancybox=True, shadow=True, ncol=5)  
         
 
 # 35000 is max tris that doesn't affect gpu, so objects with min tris could be in this area
 # 0.00779 is average of object's volume fittable on a table

 plt.axhline(y=35000,linestyle='--',color='red', xmin=0.005)
 #plt.axhline(y=100,linestyle='--',color='red', xmin=0.5)
 plt.axvline(x=0.62, ymax=log_tris,linestyle='--', color='red')

 #plt.title('Data Base Information', fontsize=11)
 plt.xlabel('Area')
 plt.ylabel('Tris')
 ax = plt.gca()
 ax.set_yscale('log')
 ax.set_xscale('log')
  
 plt.tight_layout()
 plt.savefig("database.pdf", dpi=500)
 plt.show()
 
  

    


 
'''
 sECTION FOR DATA BASE GRAPH
'''



'''

$$$$$$$$$$$$ SECTION FOR for gpu-GRAPH $$$$$$$$$$$$$$$

'''

# @@@@@@@@@@ important code for gpu- commented for now to get data for inout in above
''' 

with open('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Jan21\\extra_inf.txt', mode='r') as inform, open('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Jan21\\GPU_usage.txt', mode='r') as gp:
 
    
 GPU = gp.readlines()
 gpu2=[]
 gpu3=[]
 time=[]

 i=1
 for elm in GPU:
     gpu2= elm.split(" ")
     gpu3.append(gpu2[3])
     time.append(i)
     i+=1
 
 #time= len(GPU)
 plt.scatter(time, gpu3, marker='x')
 plt.title('Real_GPU_log')
 plt.xlabel('time')
 plt.ylabel('GPU')
 plt.show()
 
 
 inf=  inform.readlines()
 inf1=[]
 ext_inf=[]
 objname=[]
 qual_log=[]
 dist_log=[]
 deg_log=[]
 calculated_gpu_log=[]
 engblc_log=[]
 serv_req_log=[]
 object_num=[]
 i=1
 for i in range (0, len (inf)-1) : # to the number of objects
     elm= inf[i]
     inf1= elm.split(" ")
     objname.append(inf1[2])
     qual_log.append((inf1[147]))
     dist_log.append((inf1[149]))
     deg_log.append((inf1[151]))
     calculated_gpu_log.append((inf1[153]))
     engblc_log.append(float(inf1[155]))
     serv_req_log.append(float(inf1[157]))
     object_num.append(i+1)
     
     
     ### plotting obj inf
     # qlog
     currentobj_qual_log=[]
     tempobj_qual_log= (qual_log [i].split(","))
     for elm in tempobj_qual_log[0:len(tempobj_qual_log)-1]:
       currentobj_qual_log.append( float(elm))
       
     time2=[]
     for k in range (1, len(currentobj_qual_log)+1 ):
         time2.append(k)
     
     plt.scatter(time2, currentobj_qual_log, marker='x')
     plt.title('quality_log_'+ objname[i])
     plt.xlabel('time')
     plt.ylabel('quality')
     plt.show()   
     
     # distlog
     currentobj_dis_log=[]
     tempobj_dis_log= (dist_log [i].split(","))
     for elm in tempobj_dis_log[0:len(tempobj_dis_log)-1]:
       currentobj_dis_log.append( float(elm))
     
     plt.scatter(time2, currentobj_dis_log, marker='x')
     plt.title('distance_log_'+ objname[i])
     plt.xlabel('time')
     plt.ylabel('distance')
     plt.show()          
     
     # degradation_error_log
     currentobj_deg_log=[]
     tempobj_deg_log= (deg_log [i].split(","))
     for elm in tempobj_deg_log[0:len(tempobj_deg_log)-1]:
       currentobj_deg_log.append( float(elm))
     
     plt.scatter(time2, currentobj_deg_log, marker='x')
     plt.title('degradation_error_log'+ objname[i])
     plt.xlabel('time')
     plt.ylabel('degradation')
     plt.show()     
 
 ext_inf= inf[len (inf)-1].split(" ")
 
 
 
 plt.scatter(object_num, engblc_log, marker='x')
 plt.title('energy_blc_log')
 plt.xlabel('object_num')
 plt.ylabel('energy_blc')
 plt.show()
 
 plt.scatter(object_num, serv_req_log, marker='x')
 plt.title('serv_req_log')
 plt.xlabel('object_num')
 plt.ylabel('serv_req')
 plt.show()
 
 
 # now plot time 2 and logs for q, dist, ...
 
 
 
 print (ext_inf)
 
'''



'''

$$$$$$$$$$$$ SECTION FOR for gpu-GRAPH $$$$$$$$$$$$$$$

'''