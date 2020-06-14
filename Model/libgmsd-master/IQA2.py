#from __future__ import print_function
import os
import sys
import time
import argparse
import array as arr 


''' for critical view purpose
# gives you gsmd
# reads from outm2 all immages name for comparison and write results to out.txt
then read result from out to compare and find the critical view: lets say I have compared all 6 views and based on minimum value I'll write the name of object plus the critical view from 1-6 that shows one of 6 views
 '''


#python IQA.py -- --outm2 text.txt
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
with open(in3, "r") as inp, open("outIQA.txt", "w") as out:
  


  numinp=float(inp.readline())
  
  lines = inp.readlines()
  print(lines[0])
  
  i=0
  index=0
  print(numinp)
  
  
  a = []
  b = []
  name=[]
 
  while i<numinp:
     for j in range(0, 6):
        a.append (lines[i+j])
        b.append (lines[i+j+6])
        temp1=a[index]
        temp1=temp1[:-1] #trims one char at the  end to remove \n char
     
        temp2=b[index]
        temp2=temp2[:-1]
    
   
       # print(str(temp1) + " "+ str(temp2)+ " "+str(a))
      


        path=  "wine gmsd.exe"
	end=" > IQAout.txt "
        #final = "wine gmsd.exe Objects/Screenshots/Andy1d1r1.0.bmp Objects/Screenshots/Andy1d1r0.5.bmp > IQAout.txt"  
	
        final ="wine gmsd.exe %s %s > IQAout.txt" %(temp1, temp2)
        #each iqa for one comparison to be saved on iqaout and then are written to the out file 
       # print(final)
	os.system(final)
	with open("IQAout.txt", "r") as iqa:
	  #out.write(str(j+1)+" "+iqa.readline())
          out.write(iqa.readline())
        index+=1

     objname=lines[i]
     objname=objname[20:-12] #omit file address too have just name of object
     address="Objects/obj/%s.obj " %objname
     
     print(address)
     name.append (address)
     i+=12
     
print("start checking critical view")        
with open("outIQA.txt", "r") as inpp, open("cresult.txt", "w") as outt:       
   i2=0
   j2=0
   lines2 = inpp.readlines()
   while i2<numinp/2:
       maxim= max(lines2[i2:i2+6])
       maxdex= lines2.index(maxim)
       fmaxdex=maxdex%6 + 1
       print("maximum is "+ str(maxim)+ " "+str(fmaxdex))

       i2+=6
     
       outt.write(str(fmaxdex) + " " + name[j2]+"\n")
       j2+=1
''' 
    for line in inp:

        a,b = map(str,line.split())
        #out.write("%s\n" % (a + b))
	#out.write("%s\n")
	path=  "wine gmsd.exe "
	end=" > IQAout.txt "
	final= path +" "+ a +" "+ b + end
	os.system(final)
	with open("IQAout.txt", "r") as iqa:
	  out.write(iqa.readline())
'''

















'''



#comment other way
#os.system('wine gmsd.exe i25.bmp i24.bmp')
''
myCmd= 'wine gmsd.exe %in1 i24.bmp > IQAout.txt'
os.system(myCmd)

You can also store the output of the shell command in a variable in this way:

import os
myCmd = os.popen('ls -la').read()
print(myCmd)
'''










