import numpy as np 
from scipy import sparse 
from scipy.sparse.linalg import lsqr, cg, eigsh
import matplotlib.pyplot as plt 
#from matplotlib import pyplot
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import cv2
import heapq
import random
import copy
import sys
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pandas as pd
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from scipy import genfromtxt
from sklearn.cluster import SpectralClustering
from hks import *

###最適な分かれ目を見つける

def forlist(s):
    if s>0 and s<=5:
        a = np.arange(s,100,5).tolist()
        a = a[:-3]
#        a = a[:3]+a[4:-3]
        b = [0]+a+[100]
        print(b)
    else:
        print("Error")
    return b
#    これで固定
#def forlist(s):
#    if s>0 and s<=5:
#        a = np.arange(s,100,5).tolist()
#        a = a[:-3]
#        b = [0]+a+[100]
#    else:
#        print("Error")
#    return b

#kmのリストを作成
def makekmlist():
    a = [0,2,2,-2,-1,-1,5,5,5,5,5,4,4,4,4,4]
    return a


#3dオブジェクトに色付けver3
def hkscolor(inputfile, output, t=10, save = True, neigvecs=200, option = True, circle = True ):
    (VPos, VColors, ITris) = loadOffFile(inputfile)
    vnum = len(VPos)
    neigvecs = min(VPos.shape[0], neigvecs)
    hks = getHKS(VPos, ITris, neigvecs, np.array([t]))
    ris = ITris.flatten()
    if option:
#        lst = forlist(5)
#        lst = [0,8,20,30,70,100]
#        lst = [0,5,15,30,80,100]
        lst = [0,3,8,38,70,80,100]
#        lst = [0,80,100]
        binnum = len(lst[:-1])
        length = len(hks)
        index = np.argsort(hks,axis=0)
        linevpos = np.zeros((1,3))
        indexpos = np.zeros(binnum)
        
        for i in range(binnum):
            #hksの値を小さい順に並べたindexを利用して小さいものから取り出していく
            hks[index[int(lst[i]*length/100):int(lst[i+1]*length/100)]] = i
            if circle and i < binnum-1:
                line1 = np.array([])
                line2 = np.array([])
                for j in range(int((lst[i+1]-lst[i])*length/100)):
                    line1 = np.hstack((line1, np.where(ris == index[j+int(lst[i]*length/100)])[0]))
                for j in range(int((lst[i+2]-lst[i+1])*length/100)):
                    line2 = np.hstack((line2, np.where(ris == index[j+int(lst[i+1]*length/100)])[0]))
                #line1,2の境目のインデックスを取り出す
                #line1,2の中身はITris(ris)のと同じ(すなわち頂点のindex)
                #ITris(ris)の３つの頂点にline1とline2の要素２つともが含まれているもの
                line = np.intersect1d(line1//3,line2//3).astype('int32')
                hks[ITris[line].flatten()] = binnum*2
                linevpos = np.vstack((linevpos,VPos[ITris[line].flatten()]))
                indexpos[i+1] = len(linevpos)-1
        linevpos = linevpos[1:]
        indexpos = indexpos.astype('int32')
    if save:
        outputfile = output + 'time_{0}.off'.format(t)
        saveHKSColors(outputfile, VPos, hks[:, 0], ITris)
#indexposは0スタート
#hksは色付けのための変な値
#linevposは境目のVPos
#indexposはlinevposを区切るためのインデックス
#lineは境目のVPosが入っているITris
    return hks, linevpos, indexpos#, line

#his,indexposをまとめて処理
        



#def easy_hc(num,hen,km,plot=False):
def easy_hc(name,frame,hen,km,plot=False):
    time = 1000
    file = 'data/{:03d}.obj'.format(num)
    outputf = 'result/{:03d}'.format(num)
    arealength = np.zeros((17,2))
#手順1(境目の点群を抽出ファイルから読み込み)
    (vpos, clolors, itris) = loadOffFile(file)
    linevpos = np.loadtxt('hks/linvposmodel{:d}frame{:03d}.csv'.format(name,frame),delimiter=',')
    indexpos = np.loadtxt('hks/indexposmodel{:d}frame{:04d}.csv'.format(name,frame),delimiter=',').astype('int32')
#    (hks, linevpos, indexpos) = getcoloroff(file,outputf,time,False)
##############################
#手順1.5(体積が一定になるようにするsolve_volのとき）
    vol = np.loadtxt('meshvol/model{:d}meshvolframe{:04d}'.format(name,frame),delimiter=',')
    defvol = 200000
    mugni = (defvol/vol)**(1/3)
    vpos = mugni*vpos
    linevpos = mugni*linevpos
##############################
#手順2（境目の点群を各部位にわける）
#選ぶ部位 変数：hen
#kmeansを使うかどうか km = 0
#    hen = 0
#    km = 0
    kmarray = linevpos.copy()
    if km>0:
        kmeans = KMeans(n_clusters=km, init='k-means++')
        idx = kmeans.fit_predict(kmarray[indexpos[hen]:indexpos[hen+1]])
    else:
        target = (kmarray[indexpos[hen]:indexpos[hen+1]])
        (s1,s2,t) = (-target[:,2],target[:,0],target[:,1])
        aim = np.transpose(np.vstack((s1,s2,t)))
        if km==0:
            #idx0だけ異なる手法でクラスタリングを行う（重回帰分析をしてその平面より上かどうかで分ける）
            model_lr = LinearRegression()
            s = np.transpose(np.vstack((s1,s2)))
            model_lr.fit(s, t)
            u =  model_lr.coef_[0]*s1+model_lr.coef_[1]*s2+model_lr.intercept_-t
            idx = np.where(u>0,1,0)
        
####plot####
#    np.set_printoptions(threshold=len(idx))
#    print(idx)
    if plot:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlim(-0.75,0.75)
        ax.set_ylim(-0.75,0.75)
        ax.set_zlim(-0.75,0.75)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        d1 = vpos
        ax.plot(-d1[:,2],d1[:,0],d1[:,1],",")
####plot####

#手順3（主成分分析によりクラスタリングしたものの面積と周長を求める）
    d2 = linevpos[indexpos[hen]:indexpos[hen+1]]
    idxnum = np.max(idx)+1
    align = np.zeros((idxnum,3))
    for i in range(idxnum):
        (x1,x2,y) = (-d2[idx==i,2],d2[idx==i,0],d2[idx==i,1])
        aim = np.transpose(np.vstack((x1,x2,y)))
        pca = PCA(n_components=2)
        pca.fit(aim)
        result = pca.transform(aim)
        #opencv 凸包の求積
        points = result[:,:2]
        hull = ConvexHull(points)
        arealength[i] = (hull.volume,hull.area)
####plot####
        if plot:
#            comp = pca.components_
#            vec = np.cross(comp[0],comp[1])
#            vec = vec/vec[2]
#            mesh_x1 = np.arange(x1.min(), x1.max(), (x1.max()-x1.min())/20)
#            mesh_x2 = np.arange(x2.min(), x2.max(), (x2.max()-x2.min())/20)
#            mesh_x1, mesh_x2 = np.meshgrid(mesh_x1, mesh_x2)
#            (x10,x20,y0) = np.mean(aim,axis=0)
#            mesh_y = -vec[0] * (mesh_x1-x10) + -vec[1] * (mesh_x2-x20) + y0
#            ax.plot_wireframe(mesh_x1, mesh_x2, mesh_y,linewidth=0.3)
            exec("ax.plot(aim[:,0],aim[:,1],aim[:,2],marker='.',alpha = 0.3,linestyle='None',label = '%02d')"%(i))
    if plot:
        ax.legend()
####plot####
#    print('area length\n',arealength)
    return arealength






#上手い位置を探してその部分の断面積を求めるプログラム
#kmeansのところを本当はxmeansを用いたほうがいい
#plot == Trueのときは関数の外側にしたの一行を付け加える
#fig = plt.figure()#figsize=(10,10))
def solve_hc(name,frame,plot=False):
    #array = [2,3]
    #pltnum = from 1 to array[0]*array[1]
    #num = 0
    time = 700
    file = 'poseobj/model{:d}frame{:04d}.obj'.format(name,frame)
    outputf = 'result/model{:d}frame{:04d}'.format(name,frame)
#手順1(境目の点群を抽出ファイルから読み込み)
    (vpos, clolors, itris) = loadOffFile(file)
    linevpos = np.loadtxt('hks/linevposmodel{:d}frame{:04d}.csv'.format(name,frame),delimiter=',')
    indexpos = np.loadtxt('hks/indexposmodel{:d}frame{:04d}.csv'.format(name,frame),delimiter=',').astype('int32')
#    (hks, linevpos, indexpos) = getcoloroff(file,outputf,time,False)
##############################
#手順1.5(体積が一定になるようにするsolve_volのとき）
    vol = np.loadtxt('meshvol/model{:d}meshvolframe{:04d}'.format(name,frame),delimiter=',')
    defvol = 200000
    mugni = (defvol/vol)**(1/3)
    vpos = mugni*vpos
    linevpos = mugni*linevpos
##############################
#手順2（境目の点群を各部位にわける）
#選ぶ部位 変数：hen
#kmeansを使うかどうか km = 0
#hen,kmのリストを作る
#    kmlist = makekmlist()

#    kmlist  = [0,4,-1,4]
    kmlist  = [0,4,3,4]
#    kmarray = linevpos
    kmarray = linevpos.copy()
    idx = np.array([],dtype='int32')
    icount = 0 #いんっでくす調整
    notcount = []#変なやつを除くための
    for hen,km in enumerate(kmlist):
        if km>0:
            kmeans = KMeans(n_clusters=km, init='k-means++')
            kariidx = kmeans.fit_predict(kmarray[indexpos[hen]:indexpos[hen+1]])+icount
            icount += km
        else:
            target = (kmarray[indexpos[hen]:indexpos[hen+1]])
            (s1,s2,t) = (-target[:,2],target[:,0],target[:,1])
            aim = np.transpose(np.vstack((s1,s2,t)))
            if km==0:
                #idx0だけ異なる手法でクラスタリングを行う（重回帰分析をしてその平面より上かどうかで分ける）
                model_lr = LinearRegression()
                s = np.transpose(np.vstack((s1,s2)))
                model_lr.fit(s, t)
                u =  model_lr.coef_[0]*s1+model_lr.coef_[1]*s2+model_lr.intercept_-t
                kariidx = np.where(u>0,1,0)+icount
                icount += 2
            elif km==-1:#特殊，最初に2分割して足部分をさらに2分割
                kmeans = KMeans(n_clusters=3,init='k-means++')
                kmeans = KMeans(n_clusters=km, init='k-means++')
                kariidx = kmeans.fit_predict(kmarray[indexpos[hen]:indexpos[hen+1]])#+icount
                align = np.zeros((3,3))
                for i in range(3):
                    (x1,x2,y) = (d2[kariidx==i,0],d2[kariidx==i,2],d2[kariidx==i,1])
                    aim = np.transpose(np.vstack((x1,x2,y)))
                    align[i] = np.mean(aim,axis=0)
                minidx = np.argsort(align[:,2])
                target = kmarray[indexpos[hen]:indexpos[hen+1]].copy()
                kmeans = KMeans(n_clusters=3, init='k-means++')
                kariidx = kmeans.fit_predict(target)
#notcount用
                align = np.zeros((3,3))
                for i in range(3):
                    (x1,x2,y) = (-d2[kariidx==i,2],d2[kariidx==i,0],d2[kariidx==i,1])
                    aim = np.transpose(np.vstack((x1,x2,y)))
                    align[i] = np.mean(aim,axis=0)
                notcount += [np.argmax(align[:,2])+icount]
#notcount用
                kariidx += icount
                icount += 3

#        else:
#
#                kmeans = KMeans(n_clusters=2, init='k-means++')
#                kariidx = kmeans.fit_predict(kmarray[indexpos[hen]:indexpos[hen+1]])
#                d2 = linevpos[indexpos[hen]:indexpos[hen+1]]
#                align = np.zeros((2,3))
#                for i in range(2):
#                    (x1,x2,y) = (-d2[kariidx==i,2],d2[kariidx==i,0],d2[kariidx==i,1])
#                    aim = np.transpose(np.vstack((x1,x2,y)))
#                    align[i] = np.mean(aim,axis=0)
#                minidx = np.argsort(align[:,2])
#                target = kmarray[indexpos[hen]:indexpos[hen+1]].copy()
#                target[kariidx==minidx[1]] = 1
#                kmeans = KMeans(n_clusters=3, init='k-means++')
#                kariidx = kmeans.fit_predict(target)
##notcount用
#                align = np.zeros((3,3))
#                for i in range(3):
#                    (x1,x2,y) = (-d2[kariidx==i,2],d2[kariidx==i,0],d2[kariidx==i,1])
#                    aim = np.transpose(np.vstack((x1,x2,y)))
#                    align[i] = np.mean(aim,axis=0)
#                notcount += [np.argmax(align[:,2])+icount]
##notcount用
#                kariidx += icount
#                icount += 3
            elif km == -2: #用いないインデックス
                length = indexpos[hen+1]-indexpos[hen]
                kariidx = np.zeros(length)+icount
                notcount = notcount+[icount]
                icount +=1
            
#共通
        idx = np.hstack((idx,kariidx))
#    print(notcount)
####plot####
    if plot:
        #普通の体型のプロット
#        ax = fig.add_subplot(array[0],array[1],pltnum,plojection='3d')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlim(-1.0,1.0)
        ax.set_ylim(-1.0,1.0)
        ax.set_zlim(-1.0,1.0)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        d1 = vpos
        ax.plot(-d1[:,2],d1[:,0],d1[:,1],",")
####plot####

#手順3（主成分分析によりクラスタリングしたものの面積と周長を求める）
    d2 = linevpos
    align = np.zeros((icount,3))
    arealength = np.zeros((icount,2))
    for i in range(icount):
        if i not in notcount:
            (x1,x2,y) = (-d2[idx==i,2],d2[idx==i,0],d2[idx==i,1])
            aim = np.transpose(np.vstack((x1,x2,y)))
            pca = PCA(n_components=2)
            pca.fit(aim)
            result = pca.transform(aim)
            #opencv 凸包の求積
            points = result[:,:2]
            hull = ConvexHull(points)
            arealength[i] = (hull.volume,hull.area)
    ####plot####
            if plot:
    ##mesh
 #               comp = pca.components_
 #               vec = np.cross(comp[0],comp[1])
 #               vec = vec/vec[2]
 #               mesh_x1 = np.arange(x1.min(), x1.max(), (x1.max()-x1.min())/20)
 #               mesh_x2 = np.arange(x2.min(), x2.max(), (x2.max()-x2.min())/20)
 #               mesh_x1, mesh_x2 = np.meshgrid(mesh_x1, mesh_x2)
 #               (x10,x20,y0) = np.mean(aim,axis=0)
 #               mesh_y = -vec[0] * (mesh_x1-x10) + -vec[1] * (mesh_x2-x20) + y0
 #               ax.plot_wireframe(mesh_x1, mesh_x2, mesh_y,linewidth=0.3)
    ##mesh
                exec("ax.plot(aim[:,0],aim[:,1],aim[:,2],marker='.',alpha = 0.3,linestyle='None',label = '%02d')"%(i))
#    if plot:
#        ax.legend()
####plot####
#ここでnot countのところを除く
    arealength = arealength[arealength[:,0]!=0]
    return arealength

def hc_sort(array):
    lst = makekmlist()
    alsort = np.zeros((len(array),2))
    count = 0
    for i,num in enumerate(lst):
        temp=0
        if num > -2:
            if num > 0:
                temp = num
            else:
                temp = 2
            alsort[count:count+temp] = np.sort(array[count:count+temp],axis=0)
            count += temp
    return alsort

def save_saiteki(lst):
    for i,num in enumerate(lst):
        exec("al%03d = solve_hc(num)"%(num))
        exec("np.savetxt('result/saiteki/al%03d.csv',al%03d,delimiter=',')"%(num,num))
        print("\r{:.1f}%".format((i+1)/len(lst)*100), end="")

def hc_save(lst):
    ans = np.zeros((16,5))
    kmlst = makekmlist()
    count=0
    al = np.zeros((len(lst),55,2))
    alsort = np.zeros((len(lst),55,2))
    for i,num in enumerate(lst):
        al[i] = np.loadtxt('result/saiteki/al{:03d}.csv'.format(num),delimiter=',')
        alsort[i] = hc_sort(al[i])
#        exec("al%03d = np.loadtxt('result/saiteki/al{:03d}.csv'.format(num),delimiter=',')"%(num))
#        exec("alsort%03d = hc_sort(al%03d)"%(num,num))
    for a,km in enumerate(kmlst):
#temp
        temp=0
        if km > -2:
            if km > 0:
                temp = km
            else:
                temp = 2
#temp
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        print("temp is ",temp)
        if temp == 0:
            continue
        for pivot in range(len(lst)):
            sortdef = np.zeros((int(len(lst)),2))
            alsum = np.zeros((int(len(lst)),2))
            for i,j in enumerate(lst):
#                exec("alsum[i,0] = np.sum(np.abs(alsort%03d[count:count+temp,0]-alsort%03d[count:count+temp,0]))"%(lst[pivot],j))
                alsum[i,0] = np.sum(np.abs(alsort[pivot,count:count+temp,0]-alsort[i,count:count+temp,0]))
                alsum[i,1] = int(j)
            sortdef = alsum.copy()
        
            sortbox = np.zeros((20,2))
            for h in range(len(sortdef)):
                if h != pivot:
                    sortbox[int(sortdef[h,1]/10),0] += sortdef[h,0]
                    sortbox[int(sortdef[h,1]/10),1] += 1
            avebox = sortbox[:,0]/sortbox[:,1]
        
            if (int(lst[pivot]/10) in np.argsort(avebox)[:1]):#最後の数字で上位n個に正解が含まれるかどうかの確率
                count1+=1
            if (int(lst[pivot]/10) in np.argsort(avebox)[:2]):#最後の数字で上位n個に正解が含まれるかどうかの確率
                count2+=1
            if (int(lst[pivot]/10) in np.argsort(avebox)[:3]):#最後の数字で上位n個に正解が含まれるかどうかの確率
                count3+=1
            if (int(lst[pivot]/10) in np.argsort(avebox)[:4]):#最後の数字で上位n個に正解が含まれるかどうかの確率
                count4+=1
            if (int(lst[pivot]/10) in np.argsort(avebox)[:5]):#最後の数字で上位n個に正解が含まれるかどうかの確率
                count5+=1
        ans[a] = np.array([count1,count2,count3,count4,count5])/len(lst)
        count+=temp
    return ans

#frame,range,name
def matome_hc(sframe,num,name):
    time = 700
    for i in range(num):
        frame = sframe + i*10+1
        inputfile = 'poseobj/model{:d}frame{:04d}.obj'.format(name,frame)
        (h,linevpos, indexpos) = hkscolor(inputfile,None,time,False)
        np.savetxt('hks/linevposmodel{:d}frame{:04d}.csv'.format(name,frame),linevpos,delimiter=',')
        np.savetxt('hks/indexposmodel{:d}frame{:04d}.csv'.format(name,frame),indexpos,delimiter=',')
        print("\r{:.1f}%".format((i+1)/num*100), end="")
    return 0

#    for i,num in enumerate(lst):
#        inputfile = 'data/{:03d}.obj'.format(num)
#        (h,linevpos, indexpos) = hkscolor(inputfile,None,time,False)
#        exec("np.savetxt('result/time1000_tyousa/linevpos%03d.csv',linevpos,delimiter=',')"%(num))
#        exec("np.savetxt('result/time1000_tyousa/indexpos%03d.csv',indexpos,delimiter=',')"%(num))
#        print("\r{:.1f}%".format((i+1)/len(lst)*100), end="")
