#!/usr/bin/python3
import bpy
import sys
import time
import argparse
import os
import shutil
import os.path
from os import path
import time


'''decimates all objects reading object list obj_objects.txt and exports them to obj_objects/name/decimatedname.obj and decimatedname.glb I commented conversion to sfb at the end of the code for now since I no longer need sfb objects and finally to sfb_objects folder'''

'''uses these files: objects folder + obj_objects.txt as input + sfb_objects as output+ converter '''


# blender-2.80-linux-glibc217-x86_64/blender -b -P simplifier.py -- --ratio 0.5 --inm Objects/obj/andy.obj --outm Objects/obj/sandy.obj --tris tris.txt

#				command = "blender-2.80-linux-glibc217-x86_64/blender -b -P blenderSimplifyV2.py -- --ratio " + String.valueOf(perc_reduc) + " --inm ./input/" + filename + "/" + filename + ".obj --outm  ./output/" +filename + "/" + filename +perc_reduc+ "  --tris ./input/" + filename + "/tris.txt";

#this works
#blender-2.80rc2-linux-glibc217-x86_64/blender -b -P blenderSimplifyConvertObj.py --  --inm objects_obj.txt


def get_args():
  parser = argparse.ArgumentParser()
 
  # get all script args
  _, all_arguments = parser.parse_known_args()
  double_dash_index = all_arguments.index('--')
  script_args = all_arguments[double_dash_index + 1: ]
 
  # add parser rules
 # add parser rules
  #parser.add_argument('-hei', '--height', help="Ratio of reduction, Example: 0.5 mean half number of faces ")
  parser.add_argument('-in', '--inm', help="Original Model")
  
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args

 
args = get_args()

#height = float(args.height)
#print(height)

#infile = str(args.inm)




with open('objects_obj.txt', "r") as inp, open('time.txt', "+a") as outp:

 for line in inp:
   
   inpp= line[:-1]
   print (str(inpp))
   inp1= inpp.split(' ')
   decimateRatio= float(inp1[1])

   inp=inp1[0]
   print (decimateRatio)
   print('\n Clearing blender scene (default garbage...)')
   # deselect all
   bpy.ops.object.select_all(action='DESELECT')
   for o in bpy.data.objects:
            if o.type == 'MESH':
                bpy.data.objects.remove(o)

   print('\n Beginning the process of import & export using Blender Python API ...')
   bpy.ops.import_scene.obj(filepath=inp)
   print('\n Obj file imported successfully ...')
   #bpy.context.scene.objects.active = bpy.context.selected_objects[0]
   #print("poly is  "+str(len(bpy.context.object.data.polygons)))

   objname=inp[12:-4]
   print ( objname)
   
   ### just imported obj

   scene = bpy.context.scene

   obs = []
   for ob in scene.objects:
    # whatever objects you want to join...
     if ob.type == 'MESH':
         obs.append(ob)

   ctx = bpy.context.copy()

# one of the objects to join
   ctx['active_object'] = obs[0]

   ctx['selected_objects'] = obs


   bpy.ops.object.join(ctx)




#Creating a folder named as the Number of faces: named '150000'
#print('\n Creating a folder to store the decimated model ...........')
#nameOfFolder = float(args.ratio) * 100
#if not os.path.exists(str(nameOfFolder) + "%"):
#   os.makedirs(str(nameOfFolder) + "%")

#sys.exit()

   print('\n Beginning the process of Decimation using Blender Python API ...')
   modifierName='DecimateMod'

   print('\n Creating and object list and adding meshes to it ...')
   objectList=bpy.data.objects
   meshes = []
   for obj in objectList:
    if(obj.type == "MESH"):
      meshes.append(obj)

   print("{} meshes".format(len(meshes)))

   for i, obj in enumerate(meshes):
    start_time = time.time()
    obj.select_set(True)
  #ViewLayer.active=obj
    bpy.context.view_layer.objects.active = obj

    #modifier = obj.modifiers.new(modifierName,'DECIMATE')
    modifier = obj.modifiers.new(modifierName,'DECIMATE')
    modifier.ratio = decimateRatio

    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier=modifierName)

 

   output_model= "decimated"+ str(objname)+ str(decimateRatio)

# Below code works to export glb and then convert it correctly to sfb without losing the texture
   filepath="obj_objects/"+str(objname)+"/"+ output_model+'.glb'
   bpy.ops.export_scene.gltf(filepath="obj_objects/"+str(objname)+"/"+ output_model, export_materials=1 )

   ''' This is the version I used all the tim- commented ot export .glb
   filepath="obj_objects/"+str(objname)+"/"+ output_model+'.obj'
   bpy.ops.export_scene.obj(
    filepath=filepath,
    use_materials=True,  # Export materials and textures
)
   '''


   

   duration=time.time()- start_time 



   
   #duration=time.time()- start_time 
   print (" %.2f" % duration+ "  " +str(duration) + " sec")
   outp.write( str(objname) +" ratio " +str(decimateRatio) + "\n")



   print('\nProcess of Decimation Finished ...')


   home_dir = os.system("./converter --outdir ./sfb_objects/sfb_decimated"+"/ ./obj_objects/" +str(objname)+"/"+str(output_model)+".glb")# convert from glb to sfb
   #home_dir = os.system("./converter --outdir ./sfb_objects/sfb_decimated"+"/ ./obj_objects/" +str(objname)+"/"+str(output_model)+".obj")

   

