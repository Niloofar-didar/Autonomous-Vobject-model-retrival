import os
import bpy
import bmesh
import sys
import time
import argparse
import mathutils
import math
import shutil

def get_args():
  parser = argparse.ArgumentParser()
#has two outputs/:
# out.txt= name of all screenshots + all screen shots/// input is name of objects

#blender -b -P distance.py -- --infile Objects/text.txt

  # get all script args
  _, all_arguments = parser.parse_known_args()
  double_dash_index = all_arguments.index('--')
  script_args = all_arguments[double_dash_index + 1: ]
 

# add parser rules
  
  #parser.add_argument('-in', '--inm', help="Original Model")
  #parser.add_argument('-nam', '--name', help="Second Object")
  parser.add_argument('-fil', '--infile', help="Third Object")
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args
 
args = get_args()
#input_model = str(args.inm)
#print(input_model)

#
#Objname = str(args.name)
infile=str(args.infile)
print(infile)

# set initial csmera position and rotation as a reference
#bpy.data.objects['Camera'].location = mathutils.Vector((7.35, -6.92, 4.95)) #11.243 from object
        
bpy.data.objects['Camera'].rotation_euler= mathutils.Vector((1.1087, -0.0,0.8150688))
camlens =bpy.data.cameras.values()[0].lens=25


#opens in3 to read all images for compare, open out to write all results, open
 #IQAout for just one result from ubuntu


with open(infile, "r") as inp, open("Objects/Screenshots/out.txt", "w") as out:
    no_of_cases = str(inp.readline())
    print(no_of_cases)
    for line in inp:

        bpy.data.objects['Camera'].location = mathutils.Vector((7.35, -6.92, 4.95)) #11.243 from object
        input_model,Objname,Dratio,TDis = map(str,line.split())
        decimateRatio = float(Dratio)
        Dist=float(TDis) # it is preffered distance
        factor=11.243/Dist # the factor by which the camera pos should be changed
        print(factor)

        print(decimateRatio)
	# deselect all
        bpy.ops.object.select_all(action='DESELECT')
	# selection
        
        
        for o in bpy.data.objects:
            if o.type == 'MESH':
                bpy.data.objects.remove(o)

#change camera location based on the distance in input
        bpy.data.objects['Camera'].location.x /= factor
        bpy.data.objects['Camera'].location.y /=factor
        bpy.data.objects['Camera'].location.z/=factor

        camx =bpy.data.objects['Camera'].location.x
        camy =bpy.data.objects['Camera'].location.y
        camz =bpy.data.objects['Camera'].location.z
        


        print('\n Beginning the process of import using Blender Python API ...\n')
        bpy.ops.import_scene.obj(filepath=input_model)
        print('\n Obj file imported successfully ...')
	
#after importing we start simplification

        print('\n Beginning the process of Decimation using Blender Python API ...')
        modifierName='DecimateMod'
	

        print('\n Starting Distance Computing...')
        for o in bpy.data.objects:
            if o.type == 'MESH':
                o.location.x=0
                o.location.y=0
                o.location.z=0
                modifier = o.modifiers.new(modifierName,'DECIMATE')
                modifier.ratio = decimateRatio
                modifier.use_collapse_triangulate = True
                bpy.ops.object.modifier_apply(apply_as='DATA', modifier=modifierName)
                xx=o.location.x
                yy=o.location.y
                zz=o.location.z

        dx = camx - xx
        dy = camy - yy
        dz = camz - zz
        distance= math.sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2))
        print (distance)
        print('\n Ending Distance...')



	#start rotating object and screenshot
        bpy.context.scene.render.image_settings.file_format='BMP'
        output_dir="Objects/Screenshots/"
        output_file_format="bmp"
        rotation_steps = 5
        degree=0.872665
        for o in bpy.data.objects:
            if o.type == 'MESH':
        #o.rotation_euler = mathutils.Euler((90.0, -0.0, 50.0))// buttom is 150, -201, 214= 2.61799, -3.50811, 3.735 and top is 264, -115, 323= 4.60767, -2.00713, 5.63741

#5th and 6th angel
                 o.rotation_euler=mathutils.Vector((2.61799, -3.50811, 3.735))
                 bpy.context.scene.render.filepath = output_dir + Objname +str(5)+ str("d")+ str(round(distance))+ "r"+str(decimateRatio)
                 
#5th angle
                 bpy.ops.render.render(write_still = True)
                 out.write(output_dir + Objname +str(5)+ str("d")+ str(round(distance))+ "r"+str(decimateRatio)+".bmp"+"\n")

                 o.rotation_euler=mathutils.Vector((4.60767, -2.00713, 5.63741))
                 bpy.context.scene.render.filepath = output_dir + (Objname +str(6))+ "d"+ str(round(distance))+ "r"+str(decimateRatio)
                 bpy.ops.render.render(write_still = True)
                 out.write(output_dir + (Objname +str(6))+ "d"+ str(round(distance))+ "r"+str(decimateRatio)+".bmp"+"\n")
#starting 1-4th angel


                 for step in range(1, rotation_steps):
                     o.rotation_euler=mathutils.Vector((1.5708, -0.0, degree))
                     bpy.context.scene.render.filepath = output_dir + (Objname +str(step))+ "d"+ str(round(distance))+ "r"+str(decimateRatio)
                     bpy.ops.render.render(write_still = True)
                     degree+=1.5708
                     out.write(output_dir + (Objname +str(step))+ "d"+ str(round(distance))+ "r"+str(decimateRatio)+".bmp"+"\n")

#end of screenshot
  








#set the engine before rendering between (BLENDER_RENDER', 'BLENDER_GAME', 'CYCLES')
#bpy.context.scene.render.engine = 'BLENDER_RENDER'


###changing camera's distance (closer to object



#= mathutils.Euler((63.6, 0.0, 46.7= 0.815))

#euler is in radian (1.1087 = 63.5, 0.0133=0.76, 1.1483: 65.7927436)
###


#camr =bpy.data.objects['Camera'].rotation_euler
#print("rotation is "+ str(camr))

#get access to focal length

#print("camera lens is "+ str(camlens))

#print('\n Starting ...')
#print(camx)
#print(camy)
#print(camz)



#for obj in bpy.context.selected_objects:
   # bpy.context.scene.objects.active = obj
   # bpy.context.object.cycles_visibility.shadow = False

#bunny = Blender.Object.GetSelected()[0].getData()// [1] is camera , [2] is light
lamp= bpy.data.objects[2]

#print(lamp.Modes)

