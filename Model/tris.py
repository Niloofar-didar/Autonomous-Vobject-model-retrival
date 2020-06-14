#!/usr/bin/python3
import bpy
import sys
import time
import argparse

'''
This is to read object names and find the tris num, storing to another filr named tris.txt
'''
#blender -b -P tris.py -- --inm Objects/cresult.txt
 
def get_args():
  parser = argparse.ArgumentParser()
 
  # get all script args
  _, all_arguments = parser.parse_known_args()
  double_dash_index = all_arguments.index('--')
  script_args = all_arguments[double_dash_index + 1: ]
 
  # add parser rules
  parser.add_argument('-in', '--inm', help="input name of objectst")
 
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args
 
args = get_args()

input_model = str(args.inm)


with open(input_model, "r") as inp, open("Objects/tris.txt", "w") as out:
    
    for line in inp:
# Clear Blender scene
       for o in bpy.data.objects:
         if o.type == 'MESH':
           o.select = True
         else:
           o.select = False

# call the operator once
       bpy.ops.object.delete()    

       angle_num,input_model = map(str,line.split())
       

       bpy.ops.import_scene.obj(filepath=input_model)
       print('\n Obj file imported successfully ...')
       modifierName='TRIANGULATE'
       for obj in bpy.data.objects:
            if obj.type == 'MESH':
               obj.select=True
               bpy.context.scene.objects.active = obj
               objname= input_model[12:-4]
               print(" real obj name: "+objname)
               print(" name: {}".format(obj.name))
               print("{} has {} verts, {} edges, {} polys".format(obj.name, len(obj.data.vertices), len(obj.data.edges), len(obj.data.polygons)))
               modifier = obj.modifiers.new(modifierName,'TRIANGULATE')
               #modifier.ratio = 0.999999
               #modifier.use_collapse_triangulate = True
               bpy.ops.object.modifier_apply(apply_as='DATA', modifier=modifierName)
               print("{} has {} verts, {} edges, {} polys after TRIANGULATE ".format(obj.name, len(obj.data.vertices), len(obj.data.edges), len(obj.data.polygons)))

               out.write(objname + " "+str( len(obj.data.polygons))+ "\n")


        
