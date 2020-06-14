import os
import bpy
import bmesh
import sys
import time
import argparse
import mathutils
import math
import shutil
from mathutils import Quaternion

def get_args():
  parser = argparse.ArgumentParser()
#has two outputs/:
# out.txt= name of all screenshots + all screen shots/// input is name of objects

#blender -b -P distance.py -- --infile Objects/cresult.txt

  # get all script args
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

# set initial csmera position and rotation as a reference
#bpy.data.objects['Camera'].location = mathutils.Vector((7.35, -6.92, 4.95)) #11.243 from object
        
#bpy.data.objects['Camera'].rotation_euler= mathutils.Vector((1.1087, -0.0,0.8150688))
camlens =bpy.data.cameras.values()[0].lens=25
bpy.data.cameras['Camera'].lens=25



#opens in3 to read all images for compare, open out to write all results, open
 #IQAout for just one result from ubuntu

lmp=bpy.data.objects["Lamp"]
bpy.data.objects.remove(lmp)


#add new lamp

scene = bpy.context.scene

# Create new lamp datablock
lamp_data = bpy.data.lamps.new(name="Lamp2", type='HEMI')

# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="Lamp2", object_data=lamp_data)

# Link lamp object to the scene so it'll appear in this scene
scene.objects.link(lamp_object)

camera= bpy.context.scene.camera.data
camera.clip_end=1000
camera.clip_start=0.001


#Starting object z-height on camera sensror calculation:
'''
(cam-obj_Distance * obj_z_sensor) / Cam_Focal_Lenght = Real_object_z 


cam_Focal_lens= 25mm 
Real_object_z = obj.dimension.z
distance= min cam distance

then we compute virtual height of object on camera, then we change it to a fixed value of 0.458 for all objects so factor of changinf camera distance or
ratio by which we change cam distance should be virtual_z/0.458
we then will multiply cam lcation by this ratio
 
'''
#


with open(infile, "r") as inp, open("Objects/finalObj/model.txt", "w") as out:
    #no_of_cases = str(inp.readline())
    #print(no_of_cases)
    for line in inp:

        bpy.data.objects['Camera'].location = mathutils.Vector((7.35889 , -6.92579 , 4.95831 )) #11.243 from object
        
       # deselect all
        bpy.ops.object.select_all(action='DESELECT')
	# selection
        
        
        #input_model,Objname,Dratio,TDis = map(str,line.split())
        angle_num,input_model = map(str,line.split())
        angle = int( angle_num)
        objname= input_model
        objname=objname[12:-4] #omit file address too have just name of object

        print("angle is " + str(angle))
       
## flush environment before importing new object
        for o in bpy.data.objects:
            if o.type == 'MESH':
                bpy.data.objects.remove(o)
       ##

        print('\n Beginning the process of import using Blender Python API ...\n')
        bpy.ops.import_scene.obj(filepath=input_model)
        print('\n Obj file imported successfully ...')

        for o in bpy.data.objects:
            if o.type == 'MESH':
                xx=o.location.x
                yy=o.location.y
                zz=o.location.z
                curobj=o

     #nil
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

# this is min cam distance: 
        curobj.select=True # select object
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        #bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
        bpy.ops.view3d.camera_to_view_selected()
        camloc=bpy.data.objects["Camera"].location
        camlocx=str(camloc.x)
        camlocy=str(camloc.y)
        camlocz=str(camloc.z)
        
        print("cam location is "+ str(camloc))
        curobj.select=False # select object
        

#nil    






 # in order to find the fit view of object, closets distance from camera to obj that fits object, then we use the distance as a reference to calculate all dimension for the preffered distance

      
	

#start computing distance
        print('\n Starting Distance Computing...')
        camx =bpy.data.objects['Camera'].location.x
        camy =bpy.data.objects['Camera'].location.y
        camz =bpy.data.objects['Camera'].location.z


        obj= curobj
        xx=obj.location.x
        yy=obj.location.y
        zz=obj.location.z

        dx = camx - xx
        dy = camy - yy
        dz = camz - zz
        distance= math.sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2))
        inc_factor=distance #*0.5# increament distacne by this
        # so the factor to multiply cam location by each time is incremental + old_distance/ old_distance:  e.g, old-d =1 , inc = 0.5 => factor = 1 + 0.5 / 1= 1.5=> 1.5 * 1= 1.5

# next sould be 2 so factor= 1.5 + 0.5 / 1.5 =1.33 => 1.33 * old 1.5 = 2

        print ("distance is "+ str(distance))
        print('\n Ending Distance...')
        
# MAx Distance: 

        Height= curobj.dimensions.z
        #print("dz object Dimension is "+ str(curobj.dimensions.z))
        Vz= (camlens* Height)/distance
        Vx=(camlens* curobj.dimensions.x)/distance
        Vy=(camlens* curobj.dimensions.y)/distance
        
        fac1= Vx/(0.758)
        fac2= Vy/(0.758)
        fac3=Vz/(0.758)
        DFactor= (fac1+fac2+fac3)/3
        print(" Facts are " + str(fac1) +"," + str(fac2)+"," + str(fac3))
        print(" Dfactor is " + str(DFactor))
        #change camera location based on the distance in input
        '''
        Max_z= bpy.data.objects['Camera'].location.z * DFactor
        Max_y= bpy.data.objects['Camera'].location.y * DFactor
        Max_x= bpy.data.objects['Camera'].location.x * DFactor
        

        print( "Max cam location camx is "+str(Max_x )+" and actual cam locx is :"+ str(bpy.data.objects['Camera'].location.x))
        print( "Max cam location camy is "+str(Max_y )+" and actual cam locy is :"+ str(bpy.data.objects['Camera'].location.y))
        print( "Max cam location camz is "+str(Max_z )+" and actual cam locz is :"+ str(bpy.data.objects['Camera'].location.z))
        '''

        #print( "Max distance will be "+str(distance * DFactor )+" and actual Distance was  :"+ str(distance))

        Maxdistance= math.sqrt(pow(DFactor*dx, 2) + pow(DFactor*dy, 2) + pow(DFactor*dz, 2))

        #print( "Max distance by computation will be "+str(newdistance))
 
        



#after importing we start simplification
       #start rotating object, simplification  and screenshot
        bpy.context.scene.render.image_settings.file_format='BMP'
        output_dir="Objects/finalObj/"
        output_file_format="bmp"
        rotation_steps = 5

        degree=0.872665

        old_ratio=new_ratio=1
        #bpy.data.objects["Lamp"].location= bpy.data.objects["Camera"].location
        decimateRatio = float(1)
        for o in bpy.data.objects:
            if o.type == 'MESH':
              for step in range(1, 6): # 5 simplification levels screenshots
              # i=0

              # print( "max z is "+str(Max_z )+" and actual z-dim is :"+ str(bpy.data.objects['Camera'].location.z))

               bpy.data.objects["Camera"].location.x= float(camlocx)
               bpy.data.objects["Camera"].location.y= float(camlocy)
               bpy.data.objects["Camera"].location.z= float(camlocz)

               distance=0
               
               while  distance<Maxdistance :
                 bpy.data.objects["Lamp2"].location= bpy.data.objects["Camera"].location
        
                 o.rotation_euler=mathutils.Vector((1.5708, -0.0, 0.872665))
                 #i+=1

                 camxx =bpy.data.objects['Camera'].location.x
                 camyy =bpy.data.objects['Camera'].location.y
                 camzz =bpy.data.objects['Camera'].location.z
                   
                 dx = camxx - o.location.x
                 dy = camyy - o.location.y
                 dz = camzz - o.location.z
                 distance= math.sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2))
                 #print ("Distance "+str(i)+": "+str(distance))
                 #print('\n Ending Distance...')
                 
                 if angle==1:
                    degree= 0.872665
                    o.rotation_euler=mathutils.Vector((1.5708, -0.0, degree)) 
                    bpy.context.scene.render.filepath = output_dir + "deg"+ str(angle)+ "d"+ str(round(distance,2))+"r"+str(round(new_ratio,2))
                    bpy.ops.render.render(write_still = True)
                    out.write(output_dir + "deg"+ str(angle)+"d"+ str(round(distance,2))+ "r"+str(round(new_ratio,2))+ ".bmp"+"\n")

               
                 elif angle==2:
                    degree2= 0.872665+1.5708
                    o.rotation_euler=mathutils.Vector((1.5708, -0.0, degree2)) 
                    bpy.context.scene.render.filepath = output_dir +objname+ "deg"+ str(angle)+"d"+ str(round(distance,2))+ "r"+str(round(new_ratio,2))
                    bpy.ops.render.render(write_still = True)
                    out.write(output_dir +objname+ "deg"+ str(angle)+"d"+ str(round(distance,2))+ "r"+str(round(new_ratio,2))+ ".bmp"+"\n")

                
                 elif angle==3:
                    degree3= 0.872665+(2*1.5708)
                    o.rotation_euler=mathutils.Vector((1.5708, -0.0, degree3)) 
                    bpy.context.scene.render.filepath = output_dir +objname+ "deg"+ str(angle)+ "d"+ str(round(distance,2))+"r"+str(round(new_ratio,2))
                    bpy.ops.render.render(write_still = True)
                    out.write(output_dir +objname+ "deg"+ str(angle)+"d"+ str(round(distance,2))+ "r"+str(round(new_ratio,2))+ ".bmp"+"\n")

                
                 elif angle==4:
                    degree4=0.872665+(3*1.5708)
                    o.rotation_euler=mathutils.Vector((1.5708, -0.0, degree4)) 
                    bpy.context.scene.render.filepath = output_dir + "deg"+ str(angle)+ "d"+ str(round(distance,2))+"r"+str(round(new_ratio,2))
                    bpy.ops.render.render(write_still = True)
                    out.write(output_dir + objname+ "deg"+ str(angle)+"d"+ str(round(distance,2))+ "r"+str(round(new_ratio,2))+ ".bmp"+"\n")  


                 elif angle==5:
                 
                    o.rotation_euler=mathutils.Vector((2.61799, -3.50811, 3.735)) 
                    bpy.context.scene.render.filepath = output_dir+objname + "deg"+ str(angle)+ "d"+ str(round(distance,2))+"r"+str(round(new_ratio,2))
                    bpy.ops.render.render(write_still = True)
                    out.write(output_dir +objname+ "deg"+ str(angle)+"d"+ str(round(distance,2))+ "r"+str(round(new_ratio,2))+ ".bmp"+"\n")

          
                 elif angle==6:
                    o.rotation_euler=mathutils.Vector((4.60767, -2.00713, 5.63741))
                    bpy.context.scene.render.filepath = output_dir+objname + "deg"+ str(angle)+"d"+ str(round(distance,2))+ "r"+str(round(new_ratio,2))
                    bpy.ops.render.render(write_still = True)
                    out.write(output_dir+objname + "deg"+ str(angle)+"d"+ str(round(distance,2))+ "r"+str(round(new_ratio,2))+ ".bmp"+"\n")
                 
                 
                  #changinf camera location
                 mul_fact= (inc_factor+ distance) / distance

                 bpy.data.objects['Camera'].location.x *= mul_fact
                 bpy.data.objects['Camera'].location.y *=mul_fact
                 bpy.data.objects['Camera'].location.z*=mul_fact
                # print( "changed dim x, y, z is "+ str(bpy.data.objects['Camera'].location.x)+ str(bpy.data.objects['Camera'].location.y)+ str(bpy.data.objects['Camera'].location.z ))
                 
               old_ratio=new_ratio # 1 as old ratio
              # print("screen shot was taken and dec ratio was "+ str(old_ratio))
              # print('\n Beginning the process of Decimation using Blender Python API ...')
               bpy.context.scene.objects.active = o
              # print("before decimation object {} has {} verts, {} edges, {} polys".format(o.name, len(o.data.vertices), len(o.data.edges), len(o.data.polygons)))
               new_ratio=old_ratio- float(0.20) # 0.8 as new ratio
               new_ratio= round(new_ratio,2)
               decimateRatio = new_ratio/old_ratio 
               modifierName='DecimateMod'
               modifier = o.modifiers.new(modifierName,'DECIMATE')
               modifier.ratio = decimateRatio
               
               modifier.use_collapse_triangulate = True
               bpy.ops.object.modifier_apply(apply_as='DATA', modifier=modifierName)
   
               #print("After decimation object {} has {} verts, {} edges, {} polys".format(o.name, len(o.data.vertices), len(o.data.edges), len(o.data.polygons)))

               
             






        
