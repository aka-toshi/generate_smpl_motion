import bpy
import os
import math
import sys
import pandas as pd
import copy
import time
#from skinmodel import *

#skinmodelの選択肢
#IKのみ
from ikbone import *
#STRETCHのみ
#from stretchbone import *
#IK+STRETCH
#from ikstretchbone import *


def bonelen(joint):
    #配列作り
    a = np.arange(17)
    a[4] = 1
    a[7] = 1
    a[10] = 1
    a[13] = 0
    a[14] = 0
    a[-2:] -=1
    c = np.zeros(17*3,dtype='int32')
    c[::3] = a*3
    c[1::3] =a*3+1
    c[2::3] =a*3+2
    jointbox = (joint[1*3:]-joint[c])**2
    jointbox = np.reshape(jointbox,(17,3))
    jointlen = np.sum(jointbox,axis=1)**0.5
    return jointlen


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

def save_obj(path):
    path ='poseobj/'+path+'.obj'
    unselect()
    bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
    for i in bpy.data.objects:
        if i.type == 'MESH':
            name = i.name
    bpy.data.objects.get(name).select_set(True)
    bpy.ops.export_scene.obj(filepath=path)
    return

def readtrc(name):
    a = pd.read_table('trc/'+name+'.trc',header=4)
    b = a.values
    return b[:,:-1]

def boneposold(vpos,frame,name):
    bpy.ops.object.mode_set(mode='POSE')
    #骨格の取得
    amt = bpy.context.object
    #amt.name = aaaArmature
    #骨の数
    bonenum = int(len(amt.pose.bones)/2)
    #jointの回転
    lst0 = [1,0,2]
    lst1 = [1,1,-1]
    joint = vpos[frame,1:].copy()
    joint = np.reshape(joint,[18,3])
    joint = joint[:,lst0]*lst1
######################################################################################
#jointの拡大設定
    trc = np.loadtxt("trc/bonelength{}.csv".format(name[-1]),delimiter=',')
    #頭を除いた骨の数
    length = trc[frame,:13]
    print("BONE NUM IS(=13)=->>>>>>>>>>>",bonenum)
    print("BONE LENGTH SHAPE(=13)->>>>> ",np.shape(length))
#事前に作ったモデルのボーンの位置の配列
    jointpoint = np.zeros((18,3))
    jointpoint[0] = amt.pose.bones[0].head[:]
    for i in range(bonenum):
        jointpoint[i+1] =  amt.pose.bones[i+bonenum].head[:]
    lengthy = bonelen(jointpoint.flatten())[:13]
    mug = np.mean(lengthy/length)
    print('MAGNIFICATION IS :',mug)
######################################################################################
    joint = joint*mug
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


def bonepos(vpos,frame,name):
    bpy.ops.object.mode_set(mode='POSE')
    #骨格の取得
    amt = bpy.context.object
    #amt.name = aaaArmature
    #骨の数
    bonenum = int(len(amt.pose.bones)/2)
    #jointの回転
    lst0 = [1,0,2]
    lst1 = [1,1,-1]
    joint = vpos[frame,1:].copy()
    joint = np.reshape(joint,[18,3])
    joint = joint[:,lst0]*lst1
######################################################################################
#jointの拡大設定
    trc = np.loadtxt("trc/bonelength{}.csv".format(name[-1]),delimiter=',')
    #頭を除いた骨の数
    length = trc[frame,:13]
#    print("BONE NUM IS(=13)=->>>>>>>>>>>",bonenum)
#    print("BONE LENGTH SHAPE(=13)->>>>> ",np.shape(length))
#事前に作ったモデルのボーンの位置の配列
    jointpoint = np.zeros((18,3))
    jointpoint[0] = amt.pose.bones[0].head[:]
    for i in range(bonenum):
        jointpoint[i+1] =  amt.pose.bones[i+bonenum].head[:]
    lengthy = bonelen(jointpoint.flatten())[:13]
    mug = np.mean(lengthy/length)
    print('MAGNIFICATION IS :',mug)
######################################################################################
    joint = joint*mug
    #鼻の位置合わせ
    nosediff = joint[0] - amt.pose.bones[0].head[:]
    joint = joint - nosediff
    #posediff回転用
    lst2 = [0,2,1]
    lst3 = [1,-1,1]
    #実際に骨を動かす
#    for i in range(bonenum):
#        posdiff = joint[i+1]-amt.pose.bones[i+bonenum].head[:]
#        posdiff = posdiff[lst2]*lst3
#        bpy.data.objects[amt.name].pose.bones[i+bonenum].location = posdiff
    for i in range(bonenum):
        posdiff = joint[i+1]-amt.pose.bones[i+bonenum].head[:]
        posdiff = posdiff[lst2]*lst3
        bpy.data.objects[amt.name].pose.bones[i+bonenum].location = posdiff
        bone.keyframe_insert('rotation_euler', frame=i*10)
    #スティック表示

def sampleani(vpos,frame,num,name):
#    bpy.ops.object.mode_set(mode='POSE')
#    amt = bpy.context.object
#    for j in range
#    bone = bpy.data.objects[amt.name].pose.bones['VwristL']
##    bone.location[2] += 0.05
#    bone.location[2] = 0.05*i
#    bone.keyframe_insert('location', frame=i*5)
#    #スティック表示
#
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
    trc = np.loadtxt("trc/bonelength{}.csv".format(name[-1]),delimiter=',')
    length = trc[frame,:13]
    jointpoint = np.zeros((18,3))
    jointpoint[0] = amt.pose.bones[0].head[:]
    for i in range(bonenum):
        jointpoint[i+1] =  amt.pose.bones[i+bonenum].head[:]
    lengthy = bonelen(jointpoint.flatten())[:13]
    mug = np.mean(lengthy/length)
    joint = joint*mug
#    #位置合わせ別ver
#    bonemean = np.zeros(3)
#    bonemean += amt.pose.bones[0].head[:]
#    for k in range(bonenum):
#        bonemean += amt.pose.bones[k+bonenum].head[:]
#    bonemean /= (bonenum+1)
#    #位置合わせ別ver2
#    neckdiff = joint[1] -amt.pose.bones[0].tail[:]
#    joint = joint -neckdiff
    #鼻の位置合わせ
    nosediff = joint[0] - amt.pose.bones[0].head[:]
    joint = joint - nosediff
    #posediff回転用
    lst2 = [0,2,1]
    lst3 = [1,-1,1]
    #実際に骨を動かす
    #############################################追加
#    diff =
    #############################################追加
    for i in range(bonenum):
        diff = joint[i+1]-amt.pose.bones[i+bonenum].head[:]
        posdiff = diff[lst2]*lst3
        bone = bpy.data.objects[amt.name].pose.bones[i+bonenum]
        bone.location = posdiff
        #############################################追加
        if i ==0:
            amt.location = -diff
            amt.keyframe_insert('location',frame=num)
        #############################################追加
        bone.keyframe_insert('location', frame=num)
        bone.location = (0,0,0)
    #スティック表示


def main(obj_file,sframe,num):
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

    vpos = readtrc(name)

#animation 作成
    for i in range(num):
        sampleani(vpos,sframe+i,i,name)


#カメラ作成
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.camera_add()
    cam = bpy.data.objects['Camera']
    cam.location[0] = -2
    cam.location[1] = 2.5
    cam.location[2] = -0.16
    cam.rotation_euler[0] = -math.radians(50)
    cam.rotation_euler[1] = -math.radians(90)
    bpy.ops.object.light_add(type='AREA')
    light = bpy.data.objects['Area']
    light.location[0] = -2
    light.location[1] = 2.5
    light.location[2] = -0.16
    light.rotation_euler[0] = -math.radians(50)
    light.rotation_euler[1] = -math.radians(90)
    lightlight = bpy.data.lights[light.name]
    lightlight.use_shadow=False
    lightlight.energy = 100

#フレーム設定
    scene = bpy.data.scenes[0]
    scene.frame_start = 0
    scene.frame_end = i
    scene.render.filepath = '/Users/akase/img2model/exp/animation/model_'+name+'start_{:04d}'.format(sframe+1)
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.fps = 60

##アニメーション保存
#    bpy.context.scene.camera = bpy.data.objects['Camera']
#    bpy.ops.render.render(animation=True)




def solve_vol(name,frame):
#    delete_all()
#    obj_file = 'data/{:03d}.obj'.format(num)
#    bpy.ops.import_scene.obj(filepath=obj_file)
    ob = bpy.context.scene.objects[name]
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT') # Deselect all objects
#重要!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    bpy.context.view_layer.objects.active = ob   # Make the cube the active object
#重要!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ob.select_set(True)
    bpy.ops.mesh.print3d_info_volume()
    vol = np.loadtxt("meshvol.csv",delimiter=',')
    vol = np.array([vol])
    np.savetxt("meshvol/"+name+"meshvolframe{:04d}".format(frame+1),vol,delimiter=',')
    print('finish save:',name,frame)

#######################memo
#    bpy.ops.mesh.print3d_info_volume()
#について
#blenderの編集，設定，アドオンでprintと検索すると出てくるmesh:3dprint toolboxの
#ファイル位置/Applications/Blender.app/Contents/Resources/2.80/scripts/addons/object_print3d_utils
#のoperators.pyのvolumeのところを変更した
#######################memo


#~/../../Applications/Blender.app/Contents/MacOS/Blender --python default.py -- cam1.obj
if __name__ == "__main__":
    argv = sys.argv
    print(argv)
    argv = argv[argv.index("--") + 1:]
    print(argv[0],argv[1],argv[2])
    obj_file = argv[0]
    sframe = int(argv[2])
    num = int(argv[4])
    delete_all()

    main(obj_file,sframe,num)

#    # Quit Blender
#    bpy.ops.wm.quit_blender()
