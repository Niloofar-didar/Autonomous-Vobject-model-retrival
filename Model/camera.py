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

'''
gets obj and simp level, then put obj at closest dist with original and simplified quality, at 6 diff fov- screen shots , out put in 12 scresshot per each obj-> then feed to IQ2 to found critical view
'''
#blender -b -P distance.py -- --infile Objects/text.txt

  _, all_arguments = parser.parse_known_args()
  double_dash_index = all_arguments.index('--')
  script_args = all_arguments[double_dash_index + 1: ]
 

# add parser rules

  parser.add_argument('-fil', '--infile', help="Third Object")
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args
 
args = get_args()

infile=str(args.infile)
print(infile)

        
#bpy.data.objects['Camera'].rotation_euler= mathutils.Vector((1.1087, -0.0,0.8150688))
bpy.data.cameras.values()[0].lens=25
bpy.data.cameras['Camera'].lens=25



with open(infile, "r") as inp, open("Objects/Screenshots/out.txt", "w") as out:
    
    count= inp.readline()
    finalcount= float(count)*6 *2
    out.write(str(finalcount)+ "\n")

    for line in inp:

        bpy.data.objects['Camera'].location = mathutils.Vector((7.35889 , -6.92579 , 4.95831 )) #11.243 from object
        input_model,Objname,Dratio = map(str,line.split())
        
        #Dist=float(TDis) # it is preffered distance
        
        
	# deselect all
        bpy.ops.object.select_all(action='DESELECT')
	# selection
        
        
        for o in bpy.data.objects:
            if o.type == 'MESH':
                bpy.data.objects.remove(o)
           

        print('\n Beginning the process of import using Blender Python API ...\n')
        bpy.ops.import_scene.obj(filepath=input_model)
        print('\n Obj file imported successfully ...')
	
        #should be comment-for inf just
        for o in bpy.data.objects:
            if o.type == 'MESH':
                xx=o.location.x
                yy=o.location.y
                zz=o.location.z
                curobj=o

                i=1
                camx =bpy.data.objects['Camera'].location.x
                camy =bpy.data.objects['Camera'].location.y
                camz =bpy.data.objects['Camera'].location.z

                dx = camx - xx
                dy = camy - yy
                dz = camz - zz
                distance= math.sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2))
                print ("Distance "+str(i)+": "+str(distance))
                print('\n Ending Distance...')
         #should be comment-         

        
#nil
        # deselect all
        bpy.ops.object.select_all(action='DESELECT')
	# selection
        bpy.ops.object.select_pattern(pattern="Camera")
        cam=bpy.data.objects['Camera']
        tcc = cam.constraints.new(type='TRACK_TO')
        
        tcc.target = curobj
        print("current obj is "+ str(curobj))

        tcc.track_axis = "TRACK_NEGATIVE_Z"
        tcc.up_axis = 'UP_Y'

        # deselect all
        bpy.ops.object.select_all(action='DESELECT')
	# selection

#‘GEOMETRY_ORIGIN', '1-ORIGIN_GEOMETRY', 'ORIGIN_CURSOR', '2ORIGIN_CENTER_OF_MASS', 'ORIGIN_CENTER_OF_VOLUME'

        curobj.select=True # select object
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        #bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
        bpy.ops.view3d.camera_to_view_selected()
        bpy.data.objects["Lamp"].location= bpy.data.objects["Camera"].location
        camloc=bpy.data.objects["Camera"].location
        print("cam location is "+ str(camloc))
        curobj.select=False # select object

       
 
        

#nil    

	#start rotating object, simplification  and screenshot
        bpy.context.scene.render.image_settings.file_format='BMP'
        output_dir="Objects/Screenshots/"
        output_file_format="bmp"
        rotation_steps = 5

        degree=0.872665
        for o in bpy.data.objects:
            if o.type == 'MESH':

              decimateRatio = float(1)
              for step in range(1, 3):

                 print('\n Beginning the process of Decimation using Blender Python API ...')
                 modifierName='DecimateMod'
                 modifier = o.modifiers.new(modifierName,'DECIMATE')
                 modifier.ratio = decimateRatio
                 print("dec ratio is "+ str(decimateRatio))
                 modifier.use_collapse_triangulate = True
                 bpy.ops.object.modifier_apply(apply_as='DATA', modifier=modifierName)

              
                 for step in range(1, rotation_steps):
                     
                     
                     o.rotation_euler=mathutils.Vector((1.5708, -0.0, degree))
                     

                     xx=o.location.x
                     yy=o.location.y
                     zz=o.location.z
                     obj=o

                     objloc=o.location
                     print("current obj is "+ str(obj)+" and obj location is "+ str(objloc))

                     i+=1
                     camx =bpy.data.objects['Camera'].location.x
                     camy =bpy.data.objects['Camera'].location.y
                     camz =bpy.data.objects['Camera'].location.z
                   
                     dx = camx - xx
                     dy = camy - yy
                     dz = camz - zz
                     distance= math.sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2))
                     print ("Distance "+str(i)+": "+str(distance))
                     print('\n Ending Distance...')             
                   
                     bpy.context.scene.render.filepath = output_dir + (Objname +str(step))+ "d"+ str(round(distance))+ "r"+str(decimateRatio)
    
                     bpy.ops.render.render(write_still = True)
                     

                     out.write(output_dir + (Objname +str(step))+ "d"+ str(round(distance))+"r"+str(decimateRatio)+ ".bmp"+"\n")
                     degree+=1.5708
                      
                
                 
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
                 
                 decimateRatio = float(Dratio)



#end of screenshot
  




