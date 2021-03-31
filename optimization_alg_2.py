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





def QualitySelection(ind,d11):
     
    i=ind
    gamma=float(gamm[i])
    a= float(alpha[i])
    b=float(beta[i])
    c1=float(c[i])
    #print(d3)
    
    #c2= c1- ((d22** gamma)* max_d)
    #c3=c1- ((d33** gamma)* max_d)
    c1=c1-((d11** gamma)* max_d) # ax2+bx+c= (d^gamma) * max_deg


    inp1=[1.0,1.0,1.0]
    inp2=[1.0,1.0,1.0]
 
    #print("first equation")
    r1,r2=delta(a,b,c1,c[i],d11,gamma)
    r1= float(round(r1,3))
    r2=float(round(r2,3))
    #inp1.append(r1)
    inp1[0]=r1
    inp2[0]=r2
    #print("second equation")
   # r1,r2= delta(a,b,c2,c[i],d22,gamma)
    r1= float(round(r1,3))
    r2=float(round(r2,3))
    inp1[1]=r1
    inp2[1]=r2

   # r1,r2= delta(a,b,c3,c[i],d33,gamma)
    r1= float(round(r1,3))
    r2=float(round(r2,3))
    inp1[2]=r1
    inp2[2]=r2
    q1=q2=1
    
    
    #need fixing !!!
    sel_inp=0 # selects amon inp1 or inp2
    final_inp = array.array('f')
    final_inp=[0.0,0.0,0.0]
  
   
    for i in range (0,3):
        if( inp2[i]==0 or inp2[i]==1)  :
            final_inp[i]=inp1[i]
        else:
            if( inp1[i]==0 or inp1[i]==1): 
              final_inp[i]=inp2[i]
            else:
                 final_inp[i]=min (inp2[i], inp1[i])
            
            
            
    if(closer[ind]):
        qresult= adjustcloser(final_inp[0],prev_quality[ind],a,b,float(c[i]),d11,gamma)
        #qresult= adjustcloser(final_inp[0],cache[ind],a,b,float(c[i]),d11,gamma)
        GPU_usagemax=compute_GPU(1, decision_p,ind)
        q1=q2=1.0
    else:
        qresult= adjustfarther(final_inp[0],prev_quality[ind])
        #qresult= adjustfarther(final_inp[0],cache[ind])
        #GPU_usagemax=compute_GPU(prev_quality[ind], decision_p,ind)
        GPU_usagemax=compute_GPU(1, decision_p,ind)
        q1=q2=prev_quality[ind]

    '''here we have calculated gpu max for getting closer or farther from object'''

#ind=0 # current obj index
    quality=[0.0,0.0,0.0]
   
#values are 1- 'iz'' 2-'ip and delta1 or iz'  3'delta1' 4'ip and delta1
    GPU_usagedec=0
    
    final_quality=0.0
    
    '''start of status '''
    
    eb1=eb2=0.0 
    period1 = period2= ""
    '''we change eb1 for single cases and both for duplicate cases'''
    
    if (qresult=='qprev forall'): # for whole period p show i"' quality
        '''calculate gpu saving for qprev againsat q=1
        eb= saving - dec = saving'''
        quality=[prev_quality[ind],prev_quality[ind],prev_quality[ind]]
        GPU_usagedec=compute_GPU(quality[0], decision_p,ind)
        #gpumax=compute_GPU(1, decision_p,ind)
        final_quality=prev_quality[ind]
        q1=final_quality
        #eb1=
        gpusaving[ind]= GPU_usagemax-GPU_usagedec
        eb1=gpusaving[ind]
        period1="all"
        
        GPU_usagedec2=compute_GPU(q2, decision_p,ind)
        eb2=GPU_usagemax-GPU_usagedec2
        period2="all"
        #eng_dec[ind]=update_e_dec_req(ind, decision_p,final_quality ) # for object with index 0 at ti
        #eng_blc[ind]= gpusaving[ind]-eng_dec[ind]
    #closer
   
       
    elif (qresult=='iz'): # for whole period p show i"' quality
    

        iz=final_inp[0]
        if(iz==0):
            iz=1
        quality=[iz,iz,iz]
        GPU_usagedec=compute_GPU(quality[0], decision_p,ind)
        final_quality=iz
        gpusaving[ind]= GPU_usagemax-GPU_usagedec
        if(final_quality==cache[ind]):
            eng_dec[ind]=0
        else:    
         eng_dec[ind]=update_e_dec_req(ind, decision_p,final_quality ) # for object with index 0 at ti
        
        eb1= gpusaving[ind]-eng_dec[ind]
        period1="all"
        #eb1=  eng_blc[ind]
        q1=iz
        
        GPU_usagedec2=compute_GPU(q2, decision_p,ind)
        eb2=GPU_usagemax-GPU_usagedec2
        period2="all"

    elif (qresult=='cache forall'):
        GPU_usagedec=compute_GPU(cache[ind], decision_p,ind)
        #gpumax=compute_GPU(1, decision_p,ind)
        final_quality=cache[ind]
        q1=final_quality
        #eb1=
        gpusaving[ind]= GPU_usagemax-GPU_usagedec
        eb1=gpusaving[ind]
        period1="all"
    
    
    
    elif (qresult=='delta1'):
        GPU_usagedec=GPU_usagemax
        final_quality=1
        q1=1
        eb1=0
        period1="all"
        
        GPU_usagedec2=compute_GPU(q2, decision_p,ind)
        eb2=GPU_usagemax-GPU_usagedec2
        period2="all"
        
        #eng_dec[ind]=update_e_dec_req(ind, decision_p,final_quality ) # for object with index 0 at ti
        #gpusaving[ind]= GPU_usagemax-GPU_usagedec
        
        
    
        
    return float(q1),float(q2),float(eb1),float(eb2),period1,period2
    '''end of status '''

    '''
    values:
   1- qprev forall -- chck
   
   4- iz-- chck
3- cache forall
   6- delta1  --- chck
   
    
    '''
    '''having current data we are going to predict next window quality and eb'''
def predictwindow(fath,qprev,d11,ww,ind,disinterval,closer,upx,upy,prevdist,speedx,speedy ):
    
    qlog=""
    logbesteb=""
    logbestperiod= ""
    if(ww==0):
        return 0,qlog,logbesteb,logbestperiod
    
    if(ww>0):
        
        if(qprev!=1):
          cache[ind]=qprev
        else:
          cache[ind]=fath
        
        if (qprev==1):
          father=cache[ind]
        else:    
          father=qprev  
          
          
        prev_quality[ind]=qprev # assign prev quality for qualityselection func
        qual1,qual2,eb1,eb2,p1,p2= QualitySelection(ind, d11) # for current window
        
        print (str(qual1)+" "+str(qual2) + " "+str( eb1) + " "+str(eb2) + " "+str(p1) + " "+str(p2))
        ux=(speedx* decision_p) + upx 
        uy=(speedy*decision_p)+ upy
       
        '''after each decision epoch user moves up to uspee/period and we need to update distance from all objects'''
        dis=UpdateDis(prevdist, ux, uy, obj_x[ind], obj_y[ind],ind)
        dis_nextinterval=Distance((speedx* decision_p) +ux , obj_x[ind] ,(speedy*decision_p)+ uy, obj_y[ind])
        
        if(closer[ind]):
            if (dis<=dis_nextinterval):
              d1=dis
            else:
            # d3=dis
             #d1=d3-disinterval is not correct
             d1= dis_nextinterval
           #  d2=d1 + ((d3-d1)/2)
         
        else:
          d1=dis
          #d3=d1+disinterval
          #d3=dis_nextinterval
         # d2=d1 + ((d3-d1)/2)
        
        q=0
        
        eb3,qq1,eblog1, per1 =predictwindow(father,qual1,d1,ww-1,ind,disinterval,closer,ux,uy,dis,speedx,speedy )
        eb3+=eb1
        #print ("eb3 " + str(eb3))
        eb4,qq2,eblog2, per2 =predictwindow(father,qual2,d1,ww-1,ind,disinterval,closer,ux,uy,dis,speedx,speedy )
        eb4+=eb2
        #print ("eb4 " +str( eb4))
            
    if(eb3>=eb4):
        prev_quality[ind]=qual1
        best_cur_eb[ind]=eb1
        logbesteb=str(eblog1) +str(float(round (eb1,3)))+","
        logbestperiod= str(per1) +str (p1)+","
        #logbesteb=str (eb1)+", "
        #qlog=(str(round(qual1,3))+", ")
        qlog=str(qq1)+str(float(round(qual1,3)))+","
    else:
        prev_quality[ind]=qual2
        best_cur_eb[ind]=eb2
        logbesteb= str(eblog2) +str(float(round (eb2,3)))+","
        logbestperiod= str(per2) +str (p2)+","
        #logbesteb= str (eb2)+", "
        #qlog=(str(round(qual2,3))+", ")
        qlog=str(qq2)+str(float(round(qual2,3)))+","
        
    print ("eb max " +str( max(eb3,eb4)))    
    return max(eb3,eb4) , qlog, logbesteb , logbestperiod
'''before it was max (eb3,eb4) which was wrong to return and add to eng_blc since it's eng for prediction but the final correct eng for current window is t=either eb1 or eb2''
    # need to also return qual1 and qual2 each time   '''     

def Testerror(a,b,creal,d,gamma,r1):
    
     error = ((a* (r1**2)) + b*r1 + creal) / (d**gamma)
     if(error<=max_d):
         return True
     else:
         return False

def checkerror(a,b,creal,d,gamma):
    
    
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


def delta(a,b,c,c_real,d,gm):
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
      x=checkerror(a, b, c_real, d, gm) 
      #x=1
      return x1,x
    if (0.1<x2<1) :
      x=checkerror(a, b, c_real, d, gm)
      #x=1
      return x,x2
    else:
      x=checkerror(a, b, c_real, d, gm)
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
    x=checkerror(a, b, c_real, d, gm)
    #x=1
    return x,x
    #0 is delta1 here   

def Distance(x1,x2,y1,y2): 
    return math.sqrt( (y2-y1)**2 + (x2-x1)**2 )


def UpdateDis (dis, ux, uy , obj_x,obj_y,i):
    #for i in range (0,len(dis)):
        dis1= Distance(ux, obj_x ,uy, obj_y)
        if(dis1<dis):
          closer[i]=True
        else:
          closer[i]=False
        return dis1
    

def adjustcloser(x1,prevq,a,b,c,d11,gamma): # four cases we need to adjust xs to 0 or 1 which are cases with at least two '1's
 
 if (x1!=0 ): # 111 or 110
 
 
    if(abs(x1-prevq)<0.1 ):
      value='qprev forall' #means i'' for all 
     
    elif(abs(cache[ind]-x1)<0.1  and Testerror(a,b,c,d11,gamma,prevq)==True): # if we can use the prev downloaded quality instead of the closer quality
      value='cache forall' #means i'' for all 
     
    else: 
         value='iz' #means i'' for d1
   
 else:#000, 001
      value='delta1' #means delta1
     
 return value      



      


def adjustfarther(x1,prevq): # four cases we need to adjust xs to 0 or 1 which are cases with at least two '1's
 
 if (x1!=0): # 11 or 10
 
  if(abs(prevq-x1)<0.1 ):
      
      value='qprev forall' #means i'' for all 
  elif(abs(cache[ind]-x1)<0.1):
      value='cache forall'
      
  else: 
      value='iz' #means i'' for d1
         
 elif(x1==0 ): # case 100 and 101
      value='qprev forall' #means i'' for all 
   
 return value


      
#case 00x doesn't matter since 0's are already assigned and x3 is not important   
  
def compute_GPU(qual, period,i):
  '''this is to calculate gpu utilization having quality and period'''
  # we get energy gain from decimation
  #i=0
  gpu= (agpu*qual*tris[i]) + bgpu
  return gpu*period

def update_e_dec_req(ind, ti, qual):
   '''this is to update energy consumption for decimation '''  
   
   if (qual==1):
       return 0
   
   #bw =171mbit/sec for phone and net test
   bwidth= 171 
   eng_network=(229.4/bwidth) + 23.5 # constant for all objects
   
   decenergy= eng_network # energy for downloading and network constant cost
   
   '''' not important time 
   network_t=0.2 #2s constant
   tdec= t_dec[i] * qual
   t_req= t_transf[ind] + network_t +tdec
   if (t_req>ti):
     decimatable[ind]=0
   else:
    decimatable[ind]=1
   '''
   return decenergy


def pos(lst):
    return [True for x in lst if x > 0] or False    
    
def findW(upos_x,upos_y, obj_x,obj_y):
    w=1
    u_x=upos_x
    u_y=upos_y
    userfarther=False
    
    while(userfarther== False):
        unext_x=(uspeedx* decision_p) + u_x 
        unext_y=(uspeedy*decision_p)+ u_y
        
        if(upos_x<=obj_x and upos_y<=obj_y):
          if(unext_x<=obj_x and unext_y<=obj_y and upos_x<=unext_x and upos_y<= unext_y ):  # second condition should be for all if/ here I added for this to store and have, it's to make sure that user is getting closer than previous pos to avoid getting stucked in loops
             w+=1
          else:
             userfarther=True
             break
        elif(upos_x<=obj_x and upos_y>=obj_y):
          if(unext_x<=obj_x and unext_y>=obj_y):
            w+=1
          else:
             userfarther=True  
             break
        elif(upos_x>=obj_x and upos_y>=obj_y):              
          if(unext_x>=obj_x and unext_y>=obj_y):
              w+=1
          else:
             userfarther=True 
             break
             
        elif(upos_x>=obj_x and upos_y<=obj_y):      
            if(unext_x>=obj_x and unext_y<=obj_y):    
              w+=1
            else:
              userfarther=True 
              break
              
        u_x=unext_x
        u_y=unext_y
        
   
    #u_x= u_x-(uspeedx* decision_p) 
    #u_y= u_y-(uspeedy* decision_p) 
    
    return w-1,u_x,u_y  
    
        

    
    #return max(eb3,eb4)
dataset = pd.read_csv("C:\\Users\\gx7594\\OneDrive - Wayne State University\\PhD\\AR-proj Class\\Searches\\Meetings Data\\Nov\\Nov3 and 4\\degmodel_file.csv")


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

'''
agpu=4.10365E-05
bgpu=44.82908722

decision_epoch=[]

obj_count=5
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
while i<len(alpha):
 test.append(-beta[i]/(2*alpha[i]))
 i+=1
 
 
plt.scatter(upos_x, upos_y, marker='o')
x=[1,5,4,1,5]
y=[2,1,3,1,5]   

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
for i in range(0, obj_count):
  #obj_x.append(random.randint(0, 5))
  #obj_y.append(random.randint(0, 5))
  obj_x.append(x[i])
  obj_y.append(y[i])
  obj_backward.append(False)
  #min_dis.append(1)
  us_ob_dis.append(Distance(upos_x, obj_x[i], upos_y, obj_y[i]))
   
  decision_epoch.append(1)   
  totaltris+=tris[i]  
  plt.scatter(obj_x[i], obj_y[i], marker='x')
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
max_dis=10 
#after max decision visibility is false and we don't decide about an object
plt.show()

# to set y=1 for objects in FOV in decision period

while (pos(decision_epoch)): # if there exist an object where it's decision epoch is still positive
 '''for non visile objects we make decision epoch negative
 in the start of epoch for each object we decide about decimating object and computing energy consumption/ saving..
 if object is good to decimate we'll set their flag and in the end we move user one step closer/farther based on speed
 then we recalculate all object's relative epochs
  after each epoch we restart every thing --- it's better to save last quality inf''' 

 eb=0   
 ind=0
 #for ind in range(0, obj_count):
 if(ind==0)  :  
  #decision_epoch[ind]=(math.floor(us_ob_dis[ind]/mindis[ind]))  
  
  #if (decision_epoch[ind]>0 ):

    decision_p= 2
    #min_dis[ind]/uspeed
    dis_interval=uspeed/decision_p  
    
    #decision_p= min_dis[ind]/uspeed
    #dis_interval=uspeed/decision_p
    # assume that all objects are in FOV
    j=0

    #fornow=2
    max_d=max_deg[ind] * 0.2
  #d1=5.37
  #d3=8.96
 
    
 
    
    if(closer[ind]):
            
            dis_nextinterval=Distance((uspeedx* decision_p) +upos_x , obj_x[ind] ,(uspeedy*decision_p)+ upos_y, obj_y[ind])
        
           # d3=us_ob_dis[ind]
            d1=dis_nextinterval
            #d1=d3 - (dis_interval)
            #d2=d1 + ((d3-d1)/2)
         
    else:
          dis_previnterval=Distance((uspeedx* decision_p) +upos_x , obj_x[ind] ,(uspeedy*decision_p)+ upos_y, obj_y[ind])
        
          d1=us_ob_dis[ind]
          #d3=d1+dis_interval
          #d3=dis_previnterval
          #d2=d1 + ((d3-d1)/2)
          obj_backward[ind]=False
 
    closerlog[ind]+=(str(closer[ind]) + ", ")
    #max2= 0.8* max_d
    w=4
    
    
    print("inf for obj #"+ str(ind))
    
    closer1=closer[ind] 
    eb1, predictlog[ind], logbestpredicteb[ind], bestperiod1=predictwindow(cache[ind],prev_quality[ind],d1,w,ind,dis_interval,closer,upos_x,upos_y , us_ob_dis[ind],uspeedx,uspeedy)
    '''the above func predict... alters closer array so we need to restore it to default value'''
    closer[ind]=closer1
    '''first we run the prediction which fixes quality for the next period and predictates best quality between q1 and q2 while going through next w windows then i
    it returns eb of predicted plus the best quality as prevquality[ind] is set 
    now we wanna apply the best quality here again and calculate ebalance of the next period '''
    
    #distancelog[ind]+=(str(round(us_ob_dis[ind],2)) + ", ")
    
   
    '''here from closer we suppose user goes to positive x always (from left to right)'''
    
    cur_eb1=float(best_cur_eb[ind])
    cur_q=predictlog[ind].split(",")
    cur_q1=float(cur_q[len(cur_q)-2])
    #last item in predic
    prevq1=prev_quality[ind]
    flag_eb2=False
    
    
    
    if (closer[ind]and obj_backward[ind]==False):
        #blctmp=eng_blc[ind]
        #  qualtmp=quality_log[ind]
        #predictlogtmp=predictlog[ind]
        
        #check if next w window will be still in a place where when we move farther it becomes farther not closer
        newW, ux, uy= findW(upos_x,upos_y, obj_x[ind],obj_y[ind])
        
        if(newW>1):
         # ASSUME IT'S GETTing farther from upos_x + (w-1)*dis_interval to actual upos
         
         # can't go farther than newW and also more than w, in the other words
           
             
         closer[ind]=False
         newdis= Distance(ux , obj_x[ind] ,uy , obj_y[ind])# closest dis from obj
         dis_previnterval=Distance(-(uspeedx* decision_p) +ux , obj_x[ind] ,-(uspeedy*decision_p)+ uy, obj_y[ind])
         
         #cur_qq=predictlog[ind].split(",")
         #curq1=float(cur_qq[0])
         #cur_q1=float(cur_qq[0])
         
         
         
         d11=newdis
         #d33=dis_previnterval
         #d22=d11 + ((d33-d11)/2)
         
         eb2, predictlog2, logbestpredicteb2, bestperiod2 =predictwindow(1,1,d11,newW,ind,dis_interval,closer,ux,uy , newdis,-uspeedx,-uspeedy)
         
        
         cur_eb=logbestpredicteb2.split(",") # you need to find the last eb
         cur_eb2=float(cur_eb[0])
         cur_q=predictlog2.split(",")
         cur_q2=float(cur_q[0])
         prevq2=cur_q2
         
         cur_ebb1=logbestpredicteb[ind].split(",") 
        
         ebb1=ebb2=0
         if (newW<w):     
           ebb2=float(eb2)  
           for i in range(0,newW):
              ebb1+= float(cur_ebb1[newW-i])
              
             
         else:
           ebb1=float(eb1)  
           for i in range(0,w-1):
             ebb2+=float( cur_eb[i])
           if( float(cur_eb[w-1])<=0):
              ebb2+=float( cur_eb[w-1])
           else:
             j=w-1
             while (j<len(cur_eb)-1 and cur_eb[j]>=0 ):
                  j=j+1
             if(j<len(cur_eb)-1):
                  ebb2+=cur_eb[j]
         #last item in logbestpredicted2
         #cur_q2=
         #first item in predictlog2
         
         
         if(round(ebb1,3)<round(ebb2,3)):
             cur_eb1=0
             # means that we have selected from backward in posituve way that should be neg
             j=0
             while (j<len(cur_q)-1 and float(cur_eb[j])>=0 ):
                  # looking for true negative value of eb for decimated obj
                  j=j+1
                  
             if(j<len(cur_q)-1):
                  cur_eb1=float(cur_eb[j])
               
             
             prevq1=float(prevq2)
             cur_q1=float(cur_q2)
             #cur_eb1=float(cur_eb2)
             
         #else: let it be as it is
         
    
         # eb2 is useless since it is for the farthest window, so we need chain of best eb's
         closer[ind]=True
         obj_backward[ind]=True
         
         #end of if new>1
         
         #end of if closer and bckward
    
       
    
    '''upfdate everythong finally'''
    quality_log[ind]+=(str(cur_q1) + ",")     
    eng_blc[ind]+=cur_eb1
    prev_quality[ind]=prevq1
    
    
    
    if(cur_q1!=1):
      cache[ind]=cur_q1
      fthr[ind]=cur_q1
      
    else:
        cache[ind]=fthr[ind]
    
    if (decimatable[ind]==0):
        print("object can't be decimated within period p")
        decimate[ind]=0
    elif (eng_blc[ind]>0):
        decimate[ind]=1   
     
    
    
 # the end of decision period   
 upos_x=(uspeedx* decision_p) + upos_x 
 upos_y=(uspeedy*decision_p)+ upos_y
 j=0
 
 while j<obj_count: 
   '''after each decision epoch user moves up to uspee/period and we need to update distance from all objects'''
   
   distancelog[j]+=(str(round(us_ob_dis[j],2)) + ", ")  
   us_ob_dis[j]=UpdateDis(us_ob_dis[j], upos_x, upos_y, obj_x[j], obj_y[j],j)
   if(us_ob_dis[j]>max_dis and closer[j]==False):
      decision_epoch[j]=-1 # we don't want to make loop to infinity
   else:
      decision_epoch[j]=1
      
   predictlog[j]="" 
   decimate[j]=0
   decimatable[j]=0
   eng_dec[j]=0               
   gpusaving[j]=0  
   plt.scatter(upos_x, upos_y, marker='o')
   plt.scatter(obj_x, obj_y, marker='x')
   plt.show()
   j+=1

     
    
    