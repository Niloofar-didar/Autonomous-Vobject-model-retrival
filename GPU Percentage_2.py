# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 06:35:45 2020

@author: Niloofar Didar
"""
import matplotlib.gridspec as gridspec
import matplotlib.image as img
import matplotlib.pyplot as plt
import random
import numpy as np
import networkx as nx
from itertools import *
import pandas as pd  
import numpy as np  
import csv
import array as arr 
import pylab
import math 
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
dataset = pd.read_csv("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\OCT\\GPU.csv")


print("HIII")

plt.rcParams.update({'font.size': 12})

dataset.plot(x='Tris16', y='GPU16', style='o')  
plt.title('GPU vs Tris')  
plt.xlabel('Tris')  
plt.ylabel('GPU Percentage')
plt.savefig('foo.png')  
plt.show()

markers = ['>', '+',  'o', 'v', 'x', 'X', '|','p','^','<','s','p' ,'*','h','H', '+', '.',  'D',',','H','1']
#markers = ['>', '+', '.', 'o', 'v', 'x', 'X', '|','^','<','s','p','|','^','<','s','p','*','h' ,'*','h','H','1',',','H','1']
colors=[      'b', 'g',  'r', 'c', 'm', 'y', 'orange','violet','navy','blueviolet','teal','lightcoral' ,'gold','olive','deepskyblue', 'pink', 'grey',  'firebrick','deeppink','lawngreen','slategrey']
dataset2=[]
dataset2 = pd.read_csv("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\OCT\\GPU.csv",nrows=1)

with open("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\OCT\\GPU.csv") as f:
    reader = csv.reader(f)
    i = (next(reader))
    
column=len(i)    
#print(column)



X=[]
y=[]
f = plt.figure() 
stored_ind=[]
labels=[]
fig = plt.figure(figsize=(6.5,4.3))
for k in range(1, int(column/5)+1, 1):

    tris=[]
    objname=(dataset['name'+str(k)].values.reshape(-1,1))        
    tris= dataset['Tris'+str(k)]
    i=0

    
    value = random.randint(0, len(markers)-1)
    while(stored_ind.count(value)): #avoid duplicated elm
      value = random.randint(0, len(markers)-1)


    if (objname[0]!= "bigplane-expp1" and objname[0]!= "bigandy-exp1" and objname[0]!= "drawer-exp1"
         and objname[0]!="ATV-exp1"   and objname[0]!="airplane-exp1"
            and objname[0]!="bigcabin-exp1"       and objname[0]!="andyplane-exp1"
        and objname[0]!="apricot-exp1"  and objname[0]!="mixed8-exp1"
        and objname[0]!="mixed2-exp1" and objname[0]!="mixed8-exp2" and objname[0]!="mixed1-exp1"
        and objname[0]!="coca-exp1" and objname[0]!="statue-exp1" and objname[0]!="bigplant-exp1"
        
        ):
      
        
        plt.plot((dataset['Tris'+str(k)]), (dataset['GPU'+str(k)]),marker= markers[value],color=colors[value],linestyle='') 
        obj=str(objname[0])
        labels.append(obj)
        stored_ind.append(value)
    
    '''
    if (objname[0]== "exp8-simp" ):
      #plt.legend(["Mixed objects_ exp2"])
      #f = plt.figure() 
      #f.set_figwidth(4) 
      f.set_figheight(6) 
      #plt.plot(legend=None)
      plt.plot((dataset['Tris'+str(k)]), (dataset['GPU'+str(k)]),'.', color='orange')  
      
    if (objname[0]== "Big-plane-simplified"  ):
      #plt.legend(["Big-plane_exp2"])
      #plt.plot(legend=None)

      #f.set_figwidth(4) 
      f.set_figheight(6) 
      plt.plot((dataset['Tris'+str(k)]), (dataset['GPU'+str(k)]),'^', color= 'r')  
      
    if (objname[0]== "BigCabine_d3.4"  ):
      #f = plt.figure() 
      #f.set_figwidth(4) 
      f.set_figheight(6) 
        #plt.legend(["big-andy_exp1"])    
      #plt.plot(legend=None)
      plt.plot((dataset['Tris'+str(k)]), (dataset['GPU'+str(k)]),'+', color='g')  
      
    if (objname[0]== "roomtable" ):
      #plt.legend(["roomtable_exp1"])    
      #f = plt.figure() 
      #f.set_figwidth(4) 
      f.set_figheight(6) 
      #plt.plot(legend=None)
      plt.plot((dataset['Tris'+str(k)]), (dataset['GPU'+str(k)]),'s', color='blue')    
      #dataset.plot(x=(dataset['Tris'+str(k)]), y=('GPU'+str(k)), style='o')  

      '''
      
#plt.legend(["Cabine_exp1","Roomtable_exp1","Plane_exp2", "Mixed objects_ exp2" ])      
ax = plt.gca()      
#ax.legend(handles, labels)
#plt.legend(labels, loc="upper center", bbox_to_anchor=(1.26,1 ), ncol=3, fontsize=9)
plt.legend(labels, loc="upper center", bbox_to_anchor=(0.3,1 ), ncol=2, fontsize=8)

#ax.set_yscale('log')
ax.set_xscale('log')      
plt.xlabel('Total Number of Triangles on Screen')  
plt.ylabel('GPU Utilization (%)')      
plt.tight_layout()
plt.savefig("GPU_plot.pdf", dpi=500)


plt.show()

    




'''
def display_multiple_img(images, rows = 1, cols=1):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    for ind,title in enumerate(images):
        plt.imshow(images[title])
        #ax.ravel()[ind].imshow(images[title])
        #ax.ravel()[ind].set_title(title)
        #ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.savefig('Nil2.pdf', dpi=500)
    plt.show()
'''
total_images = 4
testimg3= img.imread('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\OCT\\andysmall.png')
testimg2= img.imread('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\OCT\\bike-pic.bmp')
testimg4= img.imread('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\Searches\Meetings Data\OCT\ATV-pic.bmp')
testimg1= img.imread('C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\Searches\Meetings Data\OCT\Coca-small.png')
testimage=[testimg1,testimg2,testimg3,testimg4]
#mpimg.imread('C:\Users\gx7594\OneDrive - Wayne State University\PhD\AR-proj Class\Searches\Meetings Data\OCT\cabin.jpg')

images = {'Image'+str(i): testimage[i] for i in range(total_images)}

#plt.figure(1,constrained_layout=True,figsize=(2,2))

#for i in range (0,4):
'''
plt.subplot(221)
plt.axis('off')
plt.imshow(testimage[0])

plt.subplot(222)
plt.axis('off')
plt.imshow(testimage[1])

plt.subplot(223)
plt.axis('off')
plt.imshow(testimage[2])

plt.subplot(224)
plt.axis('off')
plt.imshow(testimage[3])

plt.show()


plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.rcParams['figure.constrained_layout.use'] = True

plt.savefig("Nil.pdf", dpi=500)
#plt.subplot_tool()
'''

#fig = plt.figure(figsize=(4,2.5)) # Notice the equal aspect ratio



'''
fig1, f1_axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True)
fig2 = plt.figure(constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
f2_ax1 = fig2.add_subplot(spec2[0, 0])
fig2.imshow(testimage[i])
f2_ax2 = fig2.add_subplot(spec2[0, 1])
f2_ax3 = fig2.add_subplot(spec2[1, 0])
f2_ax4 = fig2.add_subplot(spec2[1, 1])
'''






#for i in range(0, 4):
    #plt.subplot(testimage[i])

#display_multiple_img(images, 2, 2)



'''
    
    for k in range(1, int(column/5)+1, 1):

    if(k==30):
      k+=1
    objname=(dataset['name'+str(k)].values.reshape(-1,1))
    
    plt.plot(legend=None)
    plt.plot((dataset['Tris'+str(k)]), (dataset['GPU'+str(k)]),'o')  
    #dataset.plot(x=(dataset['Tris'+str(k)]), y=('GPU'+str(k)), style='o')  
    
    #plt.title(str(objname[0]))  
    plt.xlabel('Total Number of Triangles on Screen')  
    plt.ylabel('GPU Utilization (%)')
    #plt.show()
    plt.savefig(str(objname[0])+".png")
    plt.clf()
    
'''
for i in range(1, int(column/5)+1, 1):

    if(i==30):
      i+=1
    X= ( dataset['Tris'+str(i)].values.reshape(-1,1))
    y=(dataset['GPU'+str(i)].values.reshape(-1,1))
    area_=(dataset['Area'+str(i)].values.reshape(-1,1))
    vol=(dataset['Volume'+str(i)].values.reshape(-1,1))
    name=(dataset['name'+str(i)].values.reshape(-1,1))

#Nill



    area_vol=[] # their multiplication to test for equation
    area_vol3=[] 
    area_vol2=[] 
    area_vol4=[] 
    area=[]
    volume=[]
    area2=[]
    volume2=[]
    total_tris=[]
    c=[]
    gpu0=[]

#
    j=1
    while(j<=len(y)):
    
        if((pd.isnull(vol[j-1]))):
            break
        else:
            volume2.append(float(vol[j-1]))
            area2.append(float(area_[j-1]))
            area.append(float(area_[j-1]**(1/2)))
            volume.append(float(vol[j-1])**(1/3))
            total_tris.append(float(X[j-1]))
            gpu0.append(float(y[j-1]))
            area_vol.append( float( (area2[j-1]**(1/2)) *(volume2[j-1]**(1/2))) )
            area_vol2.append(float((area[j-1]+volume[j-1]) *X[j-1]))
            area_vol3.append( float(volume[j-1] *X[j-1]) )
            area_vol4.append( float(area[j-1] *X[j-1]) ) 
            #area_vol3.append( float( ((volume[j-1]**(1/6))) *X[j-1]) )
            #area_vol4.append( float( X[j-1])) 
            c.append(1)
            j=j+1
       


    #print(volume2)
    row_count =len(total_tris)
    
    if(total_tris[len(total_tris)-1]  < 1000000):
        print (name[0] + " " + str( total_tris[len(total_tris)-1]))
    
#row_count =33
    #print(row_count)

#Z = [np.array(area),np.array(volume),np.array(total_tris),np.array(c) ]
#a sqrt(sqrt()area * qr(vol)) + ctris +d
    #Z = [np.array(area_vol),np.array(total_tris),np.array(c) ]
    #= np.array(Z).transpose()[:]

#a area + b vol + ctris +d
    #Z2 = [np.array(area2),np.array(volume2),np.array(total_tris),np.array(c)  ]
    #Za2= np.array(Z2).transpose()[:]

# a sqrt(area) * qr(vol) +b
#Z3 = [np.array(area_vol3),np.array(total_tris),np.array(c)  ]
#Za3= np.array(Z3).transpose()[:]

    Z3 = [np.array(area_vol3),np.array(c)  ]
    Za3= np.array(Z3).transpose()[:]

# a vol*area + btris + c
    Z4 = [np.array(area_vol4),np.array(c)  ]
    Za4= np.array(Z4).transpose()[:]
    
    #A*TRIS*VOL + B
    Z2 = [np.array(area_vol2),np.array(c)  ]
    Za2= np.array(Z2).transpose()[:]


    GPU = np.array(gpu0)

    #print(GPU)  
    '''
    reg = LinearRegression(fit_intercept=False).fit(Za, GPU)



    coeffs=reg.coef_.tolist()
    coeffs.append(reg.intercept_)
    print("first")
    print(coeffs)
    cof = coeffs[0]
    a1=cof[0]
    b1 = cof[1]
    c1=cof[2]


    '''
    reg2 = LinearRegression(fit_intercept=False).fit(Za2, GPU)
    coeffs2=reg2.coef_.tolist()
    coeffs2.append(reg2.intercept_)
    #print("second")
    #print(coeffs2)
    #cof2 = 
    a2=coeffs2[0]
    b2 = coeffs2[1]
    
   

    reg3 = LinearRegression(fit_intercept=False).fit(Za3, GPU)
    coeffs3=reg3.coef_.tolist()
    coeffs3.append(reg3.intercept_)
    #print("third")
    #print(coeffs3)
    #cof3 = coeffs3[0]
    a3=coeffs3[0]
    #print(a3)
    b3 = coeffs3[1]
    #c3=cof3[2]
    #why it is a problem???
    
    reg4 = LinearRegression(fit_intercept=False).fit(Za4, GPU)
    coeffs4=reg4.coef_.tolist()
    coeffs4.append(reg4.intercept_)
    #print("forth")
    #print(coeffs4)

    #coeffs4[0]
    a4=coeffs4[0]
    b4 = coeffs4[1]
    #c4=coeffs4[2]

    # = []
    model_values2 = []
    model_values3 = []
    model_values4 = []
    total_models=[]
#mse=[]
    for i in range(0, row_count, 1):
         '''
         result1 = a1 * area_vol[i] + b1 * total_tris[i] + c1
         model_values1.append(result1)
         '''
    
         result2 = a2 * area_vol2[i]+ b2
         model_values2.append(result2)
         
         #result3 = a3 * area_vol3[i] + b3 * total_tris[i] + c3
         result3 = a3 * area_vol3[i] + b3
         model_values3.append(result3)
     
         result4 = a4 *area_vol4 [i] + b4 
         model_values4.append(result4)
    
#total_models.append(model_values1) 
    total_models.append(model_values2) 
    total_models.append(model_values3) 
    total_models.append(model_values4) 

#print(total_models[0])
    perc_err=[]
    nrmse=[]
    for i in range(0, len(total_models), 1):
        mse= mean_squared_error(GPU, total_models[i] )
        rmse = np.sqrt(mse)
        nrmse.append(rmse/(100-0)) # normalized rmse
        perc_err.append(np.mean(np.abs((GPU - total_models[i]) / GPU)) * 100)
        #print("percentage error for model "+ str(i+1)+ " is " + str(perc_err[i]))
        #print('NRMSE (no intercept): for model    + {} '.format(nrmse[i]))
  
    
    file2 = open("GPU_reg2.txt","a+") 
    file2.write(str(name[0])+ " " +str(a2 )+ " "+str( b2)  + " " + str(perc_err[0] )+ " "+str(nrmse[0])+ "\n" )
    file2.close()
    
    file3 = open("GPU_reg3.txt","a+") 
#file.write(str(name[0])+ " " +str(a4 )+ " "+str( b4) + " " + str(c4) + " " + str(perc_err[3] )+ " "+str(nrmse[3])+ "\n" )
    file3.write(str(name[0])+ " " +str(a3 )+ " "+str( b3)  + " " + str(perc_err[1] )+ " "+str(nrmse[1])+ "\n" )
    file3.close()
#coeffs = [ round(elem, 4) for elem in coeffs[1] ]
    file4 = open("GPU_reg4.txt","a+") 
    file4.write(str(name[0])+ " " +str(a4 )+ " "+str( b4)  + " " + str(perc_err[2] )+ " "+str(nrmse[2])+ "\n" )
    file4.close()
#print('Multivariate Linear Regression Coefficients: ',coeffs)

    #print (name[0])
#print(round(b,4))
#c = coeffs[2]


#Nill









'''
#Affan's Code'
#print(area)

newArray = []
minValues = []
tempArray = []
percentage = []

x = 0


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
for i in newArray:
    if (i==min(minValues)):
       # print("Index of Lowest", i, min(minValues), newCounter)
        finalIndex = newCounter+2
    newCounter = newCounter + 1
    
    

newArray.reverse()



newCounter = 0
endZoneIndex = 0
for i in newArray:
    if (i==max(newArray)):
        #print("max counter", newCounter+2, i)
        endZoneIndex = newCounter+2
    newCounter = newCounter + 1


#final index value for 2nd zone has now been calculated

endZoneIndex = len(newArray) - endZoneIndex + 3
X_new = []
Y_new = []

print("Initial index", finalIndex)
print("Final index", endZoneIndex)

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



'''

def forceAspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



fig = plt.figure(figsize=(4,3))
#ax=fig    
ax = [fig.add_subplot(2,2,i+1) for i in range(4)]
lable=[  "HTris_LArea", "HTris_HArea","LowTris_LowArea" ,"LTris_HArea"]
for i in range (0,4):

    ax[i].axis('off')
    #plt.axis('off')

                
    subplot_title=(lable[i])
    ax[i].set_title(subplot_title, fontsize=10)

    #ax[i].set_title("title", fontsize= 5, loc="upper center")
    #ax[i].legend([ "HTris_LArea"])


    ax[i].imshow(testimage[i], aspect='auto')
    #if(i==0):
     #forceAspect(ax[i],aspect=0.8)
     #plt.subplots(figsize=(6, 2))




fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,       hspace = 0, wspace = 0)
    
#plt.axis('off')    
#plt.set_aspect('auto')


#plt.tight_layout()
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
#fig.subplots_adjust(wspace=0, hspace=0)
plt.gca().set_axis_off()
#plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)

plt.tight_layout()
plt.savefig("database-screen.pdf", dpi=500)
#plt.savefig('Nil2.pdf', dpi=500, bbox_inches = 'tight', pad_inches = 0)
#plt.savefig('Nil2.pdf', dpi=500, bbox_inches = 'tight', pad_inches = 0.0)
#plt.axis('off')
plt.show()

