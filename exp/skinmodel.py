import bpy
import sys
import copy
import os
import numpy as np
from mathutils import *
from math import *
#import math
#bpy.context.view_layer.update()

def rigging(name):
    joint = np.loadtxt('joint/joint'+name+'.csv',delimiter=',')
    #openposeに並び替え
    lst =[14,12,8,7,6,9,10,11,2,1,0,3,4,5,16,15,18,17,13]
    lst0 = [0,2,1]
    lst1 = np.array([1,-1,1])
    joint = joint[lst]
    joint = joint[:,lst0]*lst1


    #アーマチュアの作成
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True, location=(0,0,0))
    #作成したアーマチュアの取得
    amt = bpy.context.object
    #名前の変更
    amt.name = name+'Armature'
    #ボーンの作成
    bpy.ops.object.mode_set(mode='EDIT')
    b00 = amt.data.edit_bones.new('nose')
    b00.head = joint[0]
    b00.tail = joint[1]
    
    #親子関係の設定
    b01 = amt.data.edit_bones.new('shouldR')
    b01.head = b00.tail
    b01.tail = joint[2]

    b02 = amt.data.edit_bones.new('elbowR')
    b02.head = b01.tail
    b02.tail = joint[3]
    
    b03 = amt.data.edit_bones.new('wristR')
    b03.head = b02.tail
    b03.tail = joint[4]
    
    b04 = amt.data.edit_bones.new('shouldL')
    b04.head = b00.tail
    b04.tail = joint[5]

    b05 = amt.data.edit_bones.new('elbowL')
    b05.head = b04.tail
    b05.tail = joint[6]
    
    b06 = amt.data.edit_bones.new('wristL')
    b06.head = b05.tail
    b06.tail = joint[7]

    b07 = amt.data.edit_bones.new('hipR')
    b07.head = b00.tail
    b07.tail = joint[8]

    b08 = amt.data.edit_bones.new('kneeR')
    b08.head = b07.tail
    b08.tail = joint[9]

    b09 = amt.data.edit_bones.new('ankleR')
    b09.head = b08.tail
    b09.tail = joint[10]

    b10 = amt.data.edit_bones.new('hipL')
    b10.head = b00.tail
    b10.tail = joint[11]

    b11 = amt.data.edit_bones.new('kneeL')
    b11.head = b10.tail
    b11.tail = joint[12]

    b12 = amt.data.edit_bones.new('ankleL')
    b12.head = b11.tail
    b12.tail = joint[13]


#親子関係を付与
    b03.parent = b02
    b02.parent = b01
    b01.parent = b00
    b06.parent = b05
    b05.parent = b04
    b04.parent = b00
    b09.parent = b08
    b08.parent = b07
    b07.parent = b00
    b12.parent = b11
    b11.parent = b10
    b10.parent = b00
#関節を接続
    b03.use_connect = True
    b02.use_connect = True
    b01.use_connect = True
    b06.use_connect = True
    b05.use_connect = True
    b04.use_connect = True
    b09.use_connect = True
    b08.use_connect = True
    b07.use_connect = True
    b12.use_connect = True
    b11.use_connect = True
    b10.use_connect = True

##################ここから仮想ボーン#######################
    #オフセット
    bpos = Vector((0,0,-0.1))
    #ボーンの作成
    bv00 = amt.data.edit_bones.new('Vnose')
    bv00.head = b00.tail
    bv00.tail = b00.tail+bpos
    bv00.use_deform = False

    bv01 = amt.data.edit_bones.new('VshouldR')
    bv01.head = b01.tail
    bv01.tail = b01.tail+bpos
    bv01.use_deform = False

    bv02 = amt.data.edit_bones.new('VelbowR')
    bv02.head = b02.tail
    bv02.tail = b02.tail+bpos
    bv02.use_deform = False
    
    bv03 = amt.data.edit_bones.new('VwristR')
    bv03.head = b03.tail
    bv03.tail = b03.tail+bpos
    bv03.use_deform = False

    bv04 = amt.data.edit_bones.new('VshouldL')
    bv04.head = b04.tail
    bv04.tail = b04.tail+bpos
    bv04.use_deform = False

    bv05 = amt.data.edit_bones.new('VelbowL')
    bv05.head = b05.tail
    bv05.tail = b05.tail+bpos
    bv05.use_deform = False
    
    bv06 = amt.data.edit_bones.new('VwristL')
    bv06.head = b06.tail
    bv06.tail = b06.tail+bpos
    bv06.use_deform = False

    bv07 = amt.data.edit_bones.new('VhipR')
    bv07.head = b07.tail
    bv07.tail = b07.tail+bpos
    bv07.use_deform = False

    bv08 = amt.data.edit_bones.new('VkneeR')
    bv08.head = b08.tail
    bv08.tail = b08.tail+bpos
    bv08.use_deform = False

    bv09 = amt.data.edit_bones.new('VankleR')
    bv09.head = b09.tail
    bv09.tail = b09.tail+bpos
    bv09.use_deform = False

    bv10 = amt.data.edit_bones.new('VhipL')
    bv10.head = b10.tail
    bv10.tail = b10.tail+bpos
    bv10.use_deform = False

    bv11 = amt.data.edit_bones.new('VkneeL')
    bv11.head = b11.tail
    bv11.tail = b11.tail+bpos
    bv11.use_deform = False

    bv12 = amt.data.edit_bones.new('VankleL')
    bv12.head = b12.tail
    bv12.tail = b12.tail+bpos
    bv12.use_deform = False

############顔が動かないように
    if False:
        b13 = amt.data.edit_bones.new('eyeR')
        b13.head = b00.head
        b13.tail = joint[14]
        b14 = amt.data.edit_bones.new('eyeL')
        b14.head = b00.head
        b14.tail = joint[15]
        b15 = amt.data.edit_bones.new('earR')
        b15.head = b13.tail
        b15.tail = joint[16]
        b16 = amt.data.edit_bones.new('earL')
        b16.head = b14.tail
        b16.tail = joint[17]
        bv13 = amt.data.edit_bones.new('VeyeR')
        bv13.head = b13.tail
        bv13.tail = b13.tail+bpos
        bv13.use_deform = False
        bv14 = amt.data.edit_bones.new('VeyeL')
        bv14.head = b14.tail
        bv14.tail = b14.tail+bpos
        bv14.use_deform = False
        bv15 = amt.data.edit_bones.new('VearR')
        bv15.head = b15.tail
        bv15.tail = b15.tail+bpos
        bv15.use_deform = False
        bv16 = amt.data.edit_bones.new('VearL')
        bv16.head = b16.tail
        bv16.tail = b16.tail+bpos
        bv16.use_deform = False
        b15.parent = b13
        b13.parent = b00
        b16.parent = b14
        b14.parent = b00
        b13.use_connect = True
        b14.use_connect = True
        b15.use_connect = True
        b16.use_connect = True
    
    #骨の数
    bonenum = int(len(amt.data.edit_bones)/2)
    #レイヤーについて
    for i in range(bonenum):
        exec("bv%02d.layers[1] = True"%(i))
        exec("bv%02d.layers[0] = False"%(i))
    ##################ここまで仮想ボーン#######################

    ##ボーンコンストレイントを付与
    bpy.ops.object.mode_set(mode='POSE')
    for i in  range(bonenum):
        boneconst(i)

#表示設定
#    bpy.data.objects[amt.name].show_in_front = True
#    bpy.data.armatures[0].show_axes = True
#    bpy.data.armatures[0].show_names = True
#    bpy.data.armatures[0].layers[1] = True
#    bpy.data.armatures[0].layers[0] = False

    print("FINISH MAKE ARMATURE bone num is :",bonenum)


def boneconst(i):
#    j = i+17
    j = 'V'+bpy.context.object.pose.bones[i].name
#boneをアクティブ化する
    amt = bpy.context.object
    bone = bpy.context.object.pose.bones[i].bone
    bpy.context.object.data.bones.active =bone
#    print("active bone is",bpy.context.active_pose_bone)
#ストレッチ,IKを加える
#    bpy.ops.pose.constraint_add(type='STRETCH_TO')
    bpy.ops.pose.constraint_add(type='IK')
#ストレッチ設定
#    boneconst = bpy.data.objects[amt.name].pose.bones[i].constraints[0]
#    boneconst.target = bpy.data.objects[amt.name]
#    boneconst.subtarget = bpy.context.object.pose.bones[j].name
#    boneconst = bpy.data.objects[amt.name].pose.bones[i].constraints[1]
#IK設定
    boneconst = bpy.data.objects[amt.name].pose.bones[i].constraints[0]
    boneconst.target = bpy.data.objects[amt.name]
    boneconst.subtarget = bpy.context.object.pose.bones[j].name
    boneconst.chain_count=1
