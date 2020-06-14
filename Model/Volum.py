import bpy
#import Mathutils
import bmesh
import sys
import time
import argparse


# blender -b -P Resize.py -- --height 0.8  --inm Objects/Bed.obj --outm oBed2.obj



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
  parser.add_argument('-out', '--outm', help="Decimated output file")
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args

 
args = get_args()

#height = float(args.height)
#print(height)

input_model = str(args.inm)
print(input_model)

output_model = str(args.outm)
print(output_model)

print('\n Clearing blender scene (default garbage...)')
# deselect all
bpy.ops.object.select_all(action='DESELECT')

# selection
bpy.data.objects['Camera'].select = True

# remove it
bpy.ops.object.delete() 

# Clear Blender scene
# select objects by type
for o in bpy.data.objects:
    if o.type == 'MESH':
        o.select = True
    else:
        o.select = False

# call the operator once
bpy.ops.object.delete()

print('\n Beginning the process of import & export using Blender Python API ...')
bpy.ops.import_scene.obj(filepath=input_model)
print('\n Obj file imported successfully ...')


### just imported obj

for o in bpy.data.objects:
    if o.type == 'MESH':
        obj=o

print('\n Obj file...',obj)



def volume(obj):
    #obj.selected = True
    #bpy.ops.object.scale_apply()
    volume = 0.0
    mesh = obj.data.copy()
    mesh.transform(obj.matrix)
    for face in mesh.faces:
        if len(face.verts) > 3: return -1
        volume += mesh.verts[face.verts[0]].co.cross(mesh.verts[face.verts[1]].co).dot(mesh.verts[face.verts[2]].co)
    # blender's natural units are meters. 1m = 1bu. Imperial units use yards.
    return volume / 6 * (bpy.context.scene.unit_settings.scale_length ** 3)
    volume /= 6 * (bpy.context.scene.unit_settings.scale_length ** 3)



#self.report({'INFO'}, "Volume: %.2f" % volume)


