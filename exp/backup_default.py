import bpy
import os
import math
import sys
import pandas as pd
import copy
import time
from skinmodel import *


def delete_all():
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

def readtrc(name):
    a = pd.read_table('trc/'+name+'.trc',header=4)
    b = a.values
    return b[:,:-1]

def bonepos(vpos,frame):
    bpy.ops.object.mode_set(mode='POSE')
    #骨格の取得
    amt = bpy.context.object
    #骨の数
    bonenum = int(len(amt.pose.bones)/2)
    #jointの回転
    lst0 = [1,0,2]
    lst1 = [1,1,-1]
    joint = vpos[frame,1:].copy()
    joint = np.reshape(joint,[18,3])
    joint = joint[:,lst0]*lst1
    #jointの拡大ver1
    mug = 0.001
    joint = joint*mug
#    #jointの拡大ver2
#    X = joint[2]-joint[5]
#    lenx = np.linalg.norm(X, ord=2)
#    Y = np.array(amt.pose.bones[1+bonenum].head[:])-np.array(amt.pose.bones[4+bonenum].head[:])
#    leny = np.linalg.norm(Y, ord=2)
#    mug = leny/lenx
#    print('MAGNIFICATION IS :',mug)
#    joint = joint*mug
    #鼻の位置合わせ
    nosediff = joint[0] - amt.pose.bones[0].head[:]
    joint = joint - nosediff
    #posediff回転用
    lst2 = [0,2,1]
    lst3 = [1,-1,1]
    #実際に骨を動かす
    for i in range(bonenum):
        posdiff = joint[i+1]-amt.pose.bones[i+bonenum].head[:]
        posdiff = posdiff[lst2]*lst3
        bpy.data.objects[amt.name].pose.bones[i+bonenum].location = posdiff
    #スティック表示
    bpy.context.object.data.display_type = 'STICK'

def anim(vpos):
    framenum = 0
    for i in range(1):
        print("NOW FRAME :",i)
        frame = 1100+5*i
        print("frame",frame)
        bpy.ops.object.mode_set(mode='POSE')
        bonepos(vpos,frame)
        amt = bpy.data.objects[1]
        unselect()
        for i in amt.data.bones:
            i.select = True
        #キーフレームを打つ
        bpy.context.scene.frame_set(framenum)
#        bpy.context.object.keyframe_insert("location", group="Location")
#        bpy.context.object.keyframe_insert("scale", group="Scale")
#        bpy.context.object.keyframe_insert("rotation_quaternion", group="Roteation")
        bpy.ops.anim.keyframe_insert_menu(type = 'BUILTIN_KSI_LocRot')
        framenum+=5

def a():
    name = 'model1'
    vpos = readtrc(name)
    for i in range(10):
        bpy.context.scene.frame_set(i*5)
        bpy.ops.anim.keyframe_insert_menu(type = 'BUILTIN_KSI_LocRot')
        bonepos(vpos,1000+i*10)
    return vpos
def b():
    name = 'model1'
    vpos = readtrc(name)
    for i in range(10):
        bonepos(vpos,1010+i*10)
        time.sleep(0.5)


def main(obj_file):
    name = obj_file[:-4]
    obj_file = 'objdata/'+obj_file
    # Import object file
    bpy.ops.import_scene.obj(filepath=obj_file)
    #これで骨格を表示
    rigging(name)
    ##骨格とモデルを貼り付け
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects.get(name).select_set(True)
    bpy.data.objects.get(name+'Armature').select_set(True)
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    #trcファイルから関節位置を取り込む
    #vpos.shape = (frame,55)  (55=(time,x0,y0,z0,....,z17))
    vpos = readtrc(name)
    #フレームに合わせてポーズを変更
    bonepos(vpos,1000)

#    bpy.ops.object.mode_set(mode='POSE')
#    unselect()
#    amt = bpy.data.objects[1]
#    for i in amt.data.bones:
#        i.select = True
#    bonepos(vpos,1103)
#    anim(vpos)
#    for i in range(5):
#        bonepos(vpos,i)


#~/../../Applications/Blender.app/Contents/MacOS/Blender --python default.py -- cam1.obj
if __name__ == "__main__":
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    obj_file = argv[0]
    delete_all()

    main(obj_file)

#    # Quit Blender
#    bpy.ops.wm.quit_blender()
