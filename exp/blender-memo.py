import bpy
import math

#選択したものの色を濃くする，線を太くすることをアクティブ化と呼ぶ
#https://tamaki-py.hatenablog.com/entry/2019/05/28/180042

    # #座標軸の設定
#    for b in amt.data.edit_bones:
#        b.select = True # すべてのボーンを選択
#    bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')

def examples():
#armatureの取得
    amt = bpy.data.objects.get('myArmautre')
    amt.data.bones
    ↑bone
    amt.pose.bones
    ↑posebone

    #選択されているポーズボーンをすべて取得
    bpy.context.selected_pose_bones

    #選択したもののタイプ（class)を調べる
    print()




    #様々な例lowerspine

#    #ポーズボーンのローカル座標の取得
#    bpy.ops.object.mode_set(mode='POSE')
#    loc = amt.pose.bones['lowerspine'].location

    #回転させ方
    rot = amt.pose.bones['lowerspine'].rotation_quaternion
    #ボースボーンのアーマチュア座標の取得
    amt.pose.bones['lowerspine'].head
    #ボーズボーンのグローバル座標の取得
    #amt.matrix_world * amt.pose.bones['lowerspine'].head

    #ボーズボーンの回転モードと回転量の取得
    if amt.pose.bones['lowerspine'].rotation_mode == 'QUATERNION':
        rot = amt.pose.bones['lowerspine'].rotation_quaternion
    elif amt.pose.bones['lowerspine'].rotation_mode == 'AXIS_ANGLE':
        rot = amt.pose.bones['lowerspine'].rotation_axis_angle
    else:
        rot = amt.pose.bones['lowerspine'].rotation_euler
    
    
    #位置回転スケールのロック
    pose_bone = amt.pose.bones['lowerspine']
    pose_bone.lock_location = [True]*3
    pose_bone.lock_scale = [True]*3
    pose_bone.lock_rotation = [True]*3
    pose_bone.lock_rotation_w = True

    #ポーズボーンを隠す
    #amt.pose.bones['lowerspine'].bone.hide = True
    
    #ポーズボーンの選択
    #amt.pose.bones['lowerspine'].bone.select = True
    
    #最後に選択されたポーズボーンの取得
    #amt.pose.bones['lowerspine'].data.bones.active
    
    #IKの設定
    #c = amt.pose.bones['Bone2'].constraints.new('IK')
    #c.name = 'IK'
    #c.target = amt
    #c.subtarget = 'ik' # ik という名前のボーンを先に作成しておく
    #c.chain_count = 2
    
    #回転コピーの設定
    #c = amt.pose.bones['Bone2'].constraints.new('COPY_ROTATION')
    #c.name = 'Copy_Rotation'
    #c.target = amt
    #c.subtarget = 'lowerspine'
    #c.owner_space = 'LOCAL'
    #c.target_space = 'LOCAL'
    
    #ボーンにキーを打つ
    #b = amt.pose.bones['lowerspine']
    #b.keyframe_insert(data_path='location',group='lowerspine')
    #b.keyframe_insert(data_path='scale',group='lowerspine')
    #if b.rotation_mode == 'QUATERNION':
    #    b.keyframe_insert(data_path='rotation_quaternion',group='lowerspine')
    #elif b.rotation_mode == 'AXIS_ANGLE':
    #    b.keyframe_insert(data_path='rotation_axis_angle',group='lowerspine')
    #else:
    #    b.keyframe_insert(data_path='rotation_euler',group='lowerspine')
