import bpy
import os
import math
import sys
import numpy as np
from skinmodel import *


def delete_all():
#    bpy.ops.object.select_all(action='SELECT')
#    bpy.ops.object.delete(True)
    for i in bpy.data.objects:
        print("removeing objects", i)
        bpy.data.objects.remove(i)
    for i in bpy.data.meshes:
        print("removeing meshes", i)
        bpy.data.meshes.remove(i)
    for i in bpy.data.materials:
        print("removeing materials", i)
        bpy.data.materials.remove(i)
    for i in bpy.data.collections:
        print("removeing collections", i)
        bpy.data.collections.remove(i)
    return

#選択を全て外す
def unselect():
    for i in bpy.data.objects:
        i.select_set(False)

def import_obj(obj_file):
    bpy.ops.import_scene.obj(filepath=obj_file)

def save_obj(path='aaa'):
    unselect()
    bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
    for i in bpy.data.objects:
        if i.type == 'MESH':
            name = i.name
    bpy.data.objects.get(name).select_set(True)    
    path = path+'.obj'
    bpy.ops.export_scene.obj(filepath=path)
    return

def boneangle(num,qu,name = None):
    unselect()
    bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
    bpy.ops.object.posemode_toggle()
    for i in bpy.data.objects:
        if i.type == 'ARMATURE':
            nam = i.name
            print("selected name is -> ",nam)
    amt = bpy.data.objects.get(nam)
    b = amt.pose.bones
    if name:
        b[name].rotation_quaternion = qu
    else:
        b[num].rotation_quaternion = qu
        print("chenged angle is -> ",num,b[num].name)

    
#def solve_vol(num):
#    delete_all()
#    obj_file = 'data/{:03d}.obj'.format(num)
#    bpy.ops.import_scene.obj(filepath=obj_file)
#    ob = bpy.context.scene.objects["{:03d}".format(num)] 
#    bpy.ops.object.select_all(action='DESELECT') # Deselect all objects
##重要!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    bpy.context.view_layer.objects.active = ob   # Make the cube the active object
##重要!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    ob.select_set(True)
#    bpy.ops.mesh.print3d_info_volume()
#    print('finish',num)


#######################memo
#    bpy.ops.mesh.print3d_info_volume()
#について
#blenderの編集，設定，アドオンでprintと検索すると出てくるmesh:3dprint toolboxの
#ファイル位置/Applications/Blender.app/Contents/Resources/2.80/scripts/addons/object_print3d_utils
#のoperators.pyのvolumeのところを変更した
#######################memo


def solve_vol(name,frame):
    delete_all()
    obj_file = 'poseobj/'+name+'frame{:04d}.obj'.format(frame+1)
    bpy.ops.import_scene.obj(filepath=obj_file)
#    ob = bpy.context.scene.objects[name+'frame{:04d}.obj'.format(frame+1)] 
    ob = bpy.context.scene.objects[name]
    bpy.ops.object.select_all(action='DESELECT') # Deselect all objects
    bpy.context.view_layer.objects.active = ob   # Make the cube the active object
    ob.select_set(True)
    bpy.ops.mesh.print3d_info_volume()
    vol = np.loadtxt("meshvol.csv",delimiter=',')
    vol = np.array([vol])
#np.savetxt("meshvol/"+name+"meshvolframe{:04d}.csv".format(frame+1),vol,delimiter=',')
    np.savetxt("meshvol/"+name+"meshvolframe{:04d}".format(frame+1),vol,delimiter=',')
    print('finish save:',name,frame)


#~/../../Applications/Blender.app/Contents/MacOS/Blender --python default.py --name --frame --num
if __name__ == "__main__":
    argv = sys.argv
    print('argv is =================', argv)
    argv = argv[argv.index("--") + 1:] # get all args after "--"
    name = argv[0]
    sframe = int(argv[2])
    num = int(argv[4])


    delete_all()
    for i in range(num):
        frame = sframe+10*i
        solve_vol(name,frame)

    # Quit Blender
    bpy.ops.wm.quit_blender()
