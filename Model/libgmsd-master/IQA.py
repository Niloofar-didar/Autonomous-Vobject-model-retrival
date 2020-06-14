import os
import sys
import time
import argparse
'''
# gives you gsmd
# reads from outm2 all immages name for comparison and write results to out.txt '''
#python IQA.py -- --outm2 text.txt
def get_args():
  parser = argparse.ArgumentParser()


  _, all_arguments = parser.parse_known_args()
  double_dash_index = all_arguments.index('--')
  script_args = all_arguments[double_dash_index + 1: ]


#python IQA.py -- --inm i24.bmp --outm i25.bmp 
#myCmd= 'wine gmsd.exe i25.bmp i24.bmp > IQAout.txt'

# add parser rules
  
  #parser.add_argument('-in', '--inm', help="Original Model")
  #parser.add_argument('-out', '--outm', help="Second Object")
  parser.add_argument('-out2', '--outm2', help="Third Object")
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args
 
args = get_args()
#in1 = str(args.inm)
#in2 = str(args.outm)
in3 = str(args.outm2)
#print (in1 + " " + in2+ " "+ in3)



#opens in3 to read all images for compare, open out to write all results, open IQAout for just one result from ubuntu
with open(in3, "r") as inp, open("out.txt", "w") as out:
    #no_of_cases = str(inp.readline())
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
path=  "wine gmsd.exe "

out=" > IQAout.txt "

final= path +" "+ in1 +" "+ in2 + out

os.system(final)
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










