To generate decimated objects, please consider below guidline:

Put blenderSimplifyConvertObj.py and objects_obj.txt in a same directory, Assume that these files are in a /home/ directory, you need to make sure you have already blender-2.80rc2-linux-glibc217-x86_64 decompressed in the same directory as well as objects_obj.txt which shows the address of each object. You can find that specific version of Blender in https://download.blender.org/release/Blender2.80/

Please make sure you have all the object files (virtual objects OBJ file and the texture/image files ) stored in this directory: home/Objects/obj/object_name, as an example: home/Objects/obj/andy.obj where andy.obj is for andy object.
Then open CMD and run the command below: the bellow command, 
blender-2.80rc2-linux-glibc217-x86_64/blender -b -P blenderSimplifyConvertObj.py --  --inm objects_obj.txt

you will see all decimated objects with the format of SFB will be generated to home/sfb_objects/sfb_decimated folder. please make sure to create a folder under these names in the /home/ directory.






In this project, we create a model for assessing the user perceived quality for virtual objects based on the fact that user's ability to perceive high detail of virtual objects
degrades as user-object distance increases. Moreover, high triangle count of virtual mesh can help to improve the perceived quality of virtual object.
you can observe a sample of bunnies in below that shows both facts:

![image](https://github.com/Niloofar-didar/Autonomous-Vobject-model-retrival/assets/27611369/3d0a3c29-891a-4141-8d31-000ef2a08ca9)


In this project, we leverage IQA tool, which is to measure the degradation error of a low quality vs reference image. This tool provides an assessment very close to user-subjective assessment. Then, we integrate this tool into a framework we create levearging Blender and build a new tool to predict user perceived quality of virtual objects.
We found through our experiemnts that each object has its own chrarctristic in terms of degradation error per total trianlge count and distance.
WE model each object perceived quality through this offline tool. We put two objects of the same type, one as a reference with maximum quality and the other one with lower triangle count automatically at various distance from a virtual camera and then apply IQA to measure the second object quality relative to the first one and then use the collectd data to generate its modeling parameters.

![image](https://github.com/Niloofar-didar/Autonomous-Vobject-model-retrival/assets/27611369/5781e0de-a6c9-491b-8072-519562fbee19)

Please check the pffline profiling of virtual objects steps with a sample of Andy pobject at two different distances through three decimation ratios below:

![image](https://github.com/Niloofar-didar/Autonomous-Vobject-model-retrival/assets/27611369/a1f9a1ec-4a67-4b8b-9e69-c665cfe18be6)
