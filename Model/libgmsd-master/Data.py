#from __future__ import print_function
import os
import sys
import time
import argparse
import array as arr 


'''
# gives you distances for each object in data.txt
'''


#python IQA3.py -- --outm2 Objects/finalObj/model.txt
def get_args():
  parser = argparse.ArgumentParser()


  _, all_arguments = parser.parse_known_args()
  double_dash_index = all_arguments.index('--')
  script_args = all_arguments[double_dash_index + 1: ]




# add parser rules
  

  parser.add_argument('-out2', '--outm2', help="Third Object")
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args
 
args = get_args()

in3 = str(args.outm2)




#opens in3 to read all images for compare, open out to write all results, open IQAout for just one result from ubuntu
with open(in3, "r") as inp, open("Data.txt", "w") as out:
  


  #numinp=float(inp.readline())
  
  lines = inp.readlines()
  
  i=0
  j=0
  #while i<len(lines):
  for step in range(1,4): #num of objects
   
   if j>= len(lines):
       break 
   obj_index=j 
   out.write(str(step) + "\n")
   print j
   token=lines[obj_index]
   token=token[18:-9]
   token = token.split("deg");
   temp2 = token[1].split("d");
   obj=temp2[1] #new object distance
   out.write(str(obj) + "\n")
   temp=0 
   i+=1
   while obj!=temp:
     if i>= len(lines):
       break 
     token=lines[i]
     print i
     token=token[18:-9]
     token = token.split("deg");
     temp2=token[1]
     temp2 = temp2.split("d");
     temp=temp2[1]
     
     out.write(temp2[1]+ "\n")
     print temp2
    
     i+=1 # final i shows number of distances for each object- first one is 21
      

   j=(obj_index+ ((i-1)-obj_index)*5) # to the num of decimate 
   i=j   

  
 
  
      

         
        

        

























