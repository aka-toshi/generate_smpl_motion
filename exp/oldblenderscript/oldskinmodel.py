import bpy
import math
import sys
import copy
import os
import numpy as np

#bpy.context.view_layer.update()

def loadobjfile(filePath):
    numVertices = 0
    numUVs = 0
    numNormals = 0
    numFaces = 0
    vertices = []
    uvs = []
    normals = []
    vertexColors = []
    faceVertIDs = []
    uvIDs = []
    normalIDs = []

    for line in open(filePath, "r"):
        vals = line.split()
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            vertices.append(vals[1:4])
        if vals[0] == "f":
            fvid = []
            for f in vals[1:]:
                w = f.split("/")
                fvid.append(int(w[0]))
            faceVertIDs.append(fvid)
    vpos = np.array(vertices,dtype='float64')
    vcolor = np.array(vertexColors,dtype='float64')
    itris = np.array(faceVertIDs,dtype='int32')-1
    return vpos, vcolor, itris

def solve(name):
    (vpos, vcolor, itris) = loadobjfile(name)
    x = vpos[:,0]
    y = vpos[:,2]
    z = vpos[:,1]

    tempxp = x.copy()
    tempxm = x.copy()
    tempyp = y.copy()
    tempym = y.copy()
    tempzp = z.copy()
    tempzm = z.copy()
    xp = x>0
    xm = x<=0
    tempzp[xp] = 10
    tempzp = z - tempzp
    tempzm[xm] = 10
    tempzm = z - tempzm

    argzp = np.argsort(tempzp)[:int(len(z)/20)]
    argzm = np.argsort(tempzm)[:int(len(z)/20)]

    tempyp[argzp] = -10
    tempyp = y - tempyp
    y1 = np.argmax(tempyp)

    tempym[argzm] = -10
    tempym = y - tempym
    y2 = np.argmax(tempym)

    x1 = np.argmax(x)
    x2 = np.argmin(x)

    c = np.argmax(z)
    lst = [0,2,1]
    lst2 = [1,-1,1]

    footLtail = vpos[y1,lst]*lst2
    footRtail = vpos[y2,lst]*lst2
    headtail = vpos[c,lst]*lst2
    handLtail = vpos[x1,lst]*lst2
    handRtail = vpos[x2,lst]*lst2

    return footLtail,footRtail,headtail,handLtail,handRtail

def rigging(name):
    (footLtail,footRtail,headtail,handLtail,handRtail) = solve(name)
    #アーマチュアの作成
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True, location=(0,0,0))
    #作成したアーマチュアの取得
    amt = bpy.context.object
    #名前の変更
    amt.name = name[:-4]+'Armature'
    
    #ボーンの作成
    bpy.ops.object.mode_set(mode='EDIT')
    b = amt.data.edit_bones.new('lowerspine')
    ###############################################################気になる
    b.head = headtail*-0.3
    b.tail = headtail*0.1
    
    #デフォームの設定
    #b.use_deform = False # メッシュをデフォームしない
    b.use_deform = True 
    ###############################################################気になる
    
    #親子関係の設定
    b2 = amt.data.edit_bones.new('hipR')
    b2.head = b.head
    b2.tail = (footRtail[0],headtail[1],headtail[2]*-0.4)
    
    b3 = amt.data.edit_bones.new('hipL')
    b3.head = b.head
    b3.tail = (footLtail[0],headtail[1],headtail[2]*-0.4)
    
    b4 = amt.data.edit_bones.new('upperspine')
    b4.head = b.tail
    b4.tail = headtail*0.4
    b4.parent = b
    
    b5 = amt.data.edit_bones.new('neck')
    b5.head = b4.tail
    b5.tail = headtail*0.6
    b5.parent = b4
    
    b6 = amt.data.edit_bones.new('head')
    b6.head = b5.tail
    b6.tail = headtail
    b6.parent = b5
    
    b7 = amt.data.edit_bones.new('shoulderR')
    b7.head = b4.tail
    b7.tail = (handRtail[0]*0.2,headtail[1],handRtail[2])
    b7.parent = b4
    
    b8 = amt.data.edit_bones.new('upperarmR')
    b8.head = b7.tail
    b8.tail = (handRtail[0]*0.5,headtail[1]+0.02,handRtail[2])
    b8.parent = b7
    
    b9 = amt.data.edit_bones.new('lowerarmR')
    b9.head = b8.tail
    b9.tail = (handRtail[0]*0.8,headtail[1]+0.03,handRtail[2])
    b9.parent = b8
    
    b10 = amt.data.edit_bones.new('handR')
    b10.head = b9.tail
    b10.tail = handRtail
    b10.parent = b9
    
    b11 = amt.data.edit_bones.new('shoulderL')
    b11.head = b4.tail
    b11.tail = (handLtail[0]*0.2,headtail[1],handLtail[2])
    b11.parent = b4
    
    b12 = amt.data.edit_bones.new('upperarmL')
    b12.head = b11.tail
    b12.tail = (handLtail[0]*0.5,headtail[1]+0.02,handLtail[2])
    b12.parent = b11
    
    b13 = amt.data.edit_bones.new('lowerarmL')
    b13.head = b12.tail
    b13.tail = (handLtail[0]*0.8,headtail[1]+0.03,handLtail[2])
    b13.parent = b12
    
    b14 = amt.data.edit_bones.new('handL')
    b14.head = b13.tail
    b14.tail = handLtail
    b14.parent = b13
    
    b15 = amt.data.edit_bones.new('upperlegR')
    b15.head = b2.tail
    b15.tail = (footRtail[0],headtail[1]-0.01,headtail[2]*-1.2)
    b15.parent = b2
    
    b16 = amt.data.edit_bones.new('lowerlegR')
    b16.head = b15.tail
    b16.tail = (footRtail[0],headtail[1]+0.04,footLtail[2]+0.04)
    b16.parent = b15
    
    b17 = amt.data.edit_bones.new('footR')
    b17.head = b16.tail
    b17.tail = footRtail
    b17.parent = b16
    
    b18 = amt.data.edit_bones.new('upperlegL')
    b18.head = b3.tail
    b18.tail = (footLtail[0],headtail[1]-0.01,headtail[2]*-1.2)
    b18.parent = b3
    
    b19 = amt.data.edit_bones.new('lowerlegL')
    b19.head = b18.tail
    b19.tail = (footLtail[0],headtail[1]+0.04,footLtail[2]+0.04)
    b19.parent = b18
    
    b20 = amt.data.edit_bones.new('footL')
    b20.head = b19.tail
    b20.tail = footLtail
    b20.parent = b19
    
#    #座標軸の設定
#    for b in amt.data.edit_bones:
#        b.select = True # すべてのボーンを選択
#    bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')
