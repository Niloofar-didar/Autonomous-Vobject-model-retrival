#from __future__ import print_function
import os
import sys
import time
import argparse
import array as arr 


'''
# gives you gsmd
# reads from outm2 all immages name for comparison and write results to out.txt
'''


#python IQA3.py -- --outm2 Objects/finalObj/model.txt
def get_args():
  parser = argparse.ArgumentParser()


  _, all_arguments = parser.parse_known_args()
  double_dash_index = all_arguments.index('--')
  script_args = all_arguments[double_dash_index + 1: ]


#python IQA.py -- --inm i24.bmp --outm i25.bmp 
#myCmd= 'wine gmsd.exe i25.bmp i24.bmp > IQAout.txt'

# add parser rules
  

  parser.add_argument('-out2', '--outm2', help="Third Object")
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args
 
args = get_args()

in3 = str(args.outm2)




#opens in3 to read all images for compare, open out to write all results, open IQAout for just one result from ubuntu
with open(in3, "r") as inp, open("FinalIQA.txt", "w") as out:
  


  #numinp=float(inp.readline())
  
  lines = inp.readlines()
  
  i=0
  index=0
  
  '''
  alphabet = "a b c d e f g"
  data = alphabet.split(" ",2) #maxsplit

  for temp in data:
      print temp
  '''
  token=lines[0]
  token=token[35:-5]
  ind=0
  token3=token2=token #token2 is always fixed to first object value/ ratio num
  object_index=ind

   #for step in range(1,4):


    while token2==token3:

     token=lines[i]
     print i
     token=token[34:-5]
     token = token.split("r");
     for temp in token:
      if temp !="":
       if float(temp)<=1.0 :
         token3= temp
         print token3

     i+=1 # final i shows number of distances for each object- first one is 21
    
  
 
    for j in range(0,i-1): 
      b=[]
      a = lines[j]
      a=a[:-1]
      temp2=lines[i-1+j]
      temp2=temp2[:-1]
      b.append (temp2)
      temp2=lines[2*(i-1)+j]
      temp2=temp2[:-1]
      b.append (temp2)
      temp2=lines[3*(i-1)+j]
      temp2=temp2[:-1]
      b.append (temp2)
      temp2=lines[4*(i-1)+j]
      temp2=temp2[:-1]
      b.append (temp2)
      print a + "\n"
      print b 
      print "\n"
      
      step=0
      while(step!=4):

         final ="wine gmsd.exe %s %s > Final.txt" %(a, b[step])
         os.system(final)
         with open("Final.txt", "r") as iqa:
          out.write(iqa.readline())

         step+=1
      
  
























