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

def makeLaplacianMatrixUmbrellaWeights(VPos, ITris, anchorsIdx = [], anchorWeights = 1):
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.ones(M*6)
    for shift in range(3):
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j]
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i]
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    L[L > 0] = 1
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L
    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    V = L.data.tolist()
    I = I + list(range(N, N+len(anchorsIdx)))
    J = J + anchorsIdx
    V = V + [anchorWeights]*len(anchorsIdx)
    L = sparse.coo_matrix((V, (I, J)), shape=(N+len(anchorsIdx), N)).tocsr()
    return L

def getHKS(VPos, ITris, K, ts):
    L = makeLaplacianMatrixUmbrellaWeights(VPos, ITris)
    (eigvalues, eigvectors) = eigsh(L, K, which='LM', sigma = 0)
    res = (eigvectors[:, :, None]**2)*np.exp(-eigvalues[None, :, None]*ts.flatten()[None, None, :])
    return np.sum(res, 1)

def loadOffFile(fliePath):
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
    for line in open(fliePath, "r"):
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
    return (np.array(vertices,dtype='float64'),np.array(vertexColors,dtype='float64'),np.array(faceVertIDs,dtype='int32')-1)

def saveOffFile(filename, VPos, VColors, ITris):
    nV = VPos.shape[0]
    nF = ITris.shape[0]
    fout = open(filename, "w")
    if VColors.size == 0:
        fout.write("OFF\n%i %i %i\n"%(nV, nF, 0))
    else:
        fout.write("COFF\n%i %i %i\n"%(nV, nF, 0))
    for i in range(nV):
        fout.write("%g %g %g"%tuple(VPos[i, :]))
        if VColors.size > 0:
            fout.write(" %g %g %g"%tuple(VColors[i, :]))
        fout.write("\n")
    for i in range(nF):
        fout.write("3 %i %i %i\n"%tuple(ITris[i, :]))
    fout.close()

def saveHKSColors(filename, VPos, hks, ITris, cmap = 'plasma'):
    c = plt.get_cmap(cmap)
    x = (hks - np.min(hks))
    x /= np.max(x)
    np.array(np.round(x*255.0), dtype=np.int32)
    C = c(x)
    C = C[:, 0:3]
    saveOffFile(filename, VPos, C, ITris)

def make_hist(inputfile, output, timecount, width, start, neigvecs=200, save = False):
    (VPos, VColors, ITris) = loadOffFile(inputfile)
    vnum = len(VPos)
    heatsd = np.zeros((vnum, timecount))
    hsd_hist = np.zeros((2,vnum*timecount))#時間付き
    neigvecs = min(VPos.shape[0], neigvecs)
    for i in range(timecount):
        t = i*width + start
        hks = getHKS(VPos, ITris, neigvecs, np.array([t]))
        heatsd[:,i] = np.reshape(hks, vnum)
        hsd_hist[0,vnum*i:vnum*(i+1)] = np.reshape(hks,vnum)
        hsd_hist[1,vnum*i:vnum*(i+1)] = np.full(vnum,t)
        if save:
            outputfile = output + '{0}.obj'.format(t)
            saveHKSColors(outputfile, VPos, hks[:, 0], ITris)
        print("\r{:.1f}%".format((i+1)/timecount*100), end="")
    print("")
    return heatsd, hsd_hist

#heatsdを取得する（ヒストグラムの元）
def getheatsd(inputf, timecount, width, start):
    inputfile = 'data/{0}.obj'.format(inputf)
    print(inputfile)
    outputf = 'result'
    outputfile = 'data/color{0}'.format(inputf)
    heatsd, hsd_hist= make_hist(inputfile, outputfile, timecount, width, start)
    np.savetxt('{0}/heatsd_{1}c{2}w{3}s{4}.csv'.format(outputf, inputf, timecount, int(width), int(start)),heatsd, delimiter=',')
    return heatsd

#データからヒストグラムの元を生成(heatsdからプロットする)
#getheatsdの値に合わせる
#このhistは正規化済（頂点数に関係しない)
def histtoplot(filename, timecount, width, start):
    heatsd = np.loadtxt('result/heatsd_{0}c{1}w{2}s{3}.csv'.format(filename,timecount,width,start),delimiter=',')
#    heatsd = np.loadtxt('result/heatsd_off/heatsd_{0}c{1}w{2}s{3}.csv'.format(filename,timecount,width,start),delimiter=',')
    b = 100 #binの数
    hist = np.zeros((timecount,b))
    length = len(heatsd[:,0])
    for i in range(timecount):
        n = np.min(heatsd[:,i])
        x = np.max(heatsd[:,i])
        minn = n
        maxx = n+(x-n)/1
        hist[i] = np.histogram(heatsd[:,i], bins = b, range = (minn, maxx))[0]/length
    return hist

#点群の最小内包球求め方
def solvesph(point):
    move = 0.5
    center = np.average(point,axis=0)
    while(move>1.0e-7):
        for i in range(100):
            temp = np.sum((point-center)**2,axis=1)
            index = np.argmax(temp)
            center += (point[index]-center)*move
            radian = np.sqrt(temp[index])
        move /= 2.0
    return radian, center


#3dオブジェクトに色付け
def hkscolor(inputfile, output, t=10, neigvecs=200, save = True):
    (VPos, VColors, ITris) = loadOffFile(inputfile)
    vnum = len(VPos)
    neigvecs = min(VPos.shape[0], neigvecs)
    hks = getHKS(VPos, ITris, neigvecs, np.array([t]))
    if save:
        outputfile = output + 'time_{0}.off'.format(t)
        saveHKSColors(outputfile, VPos, hks[:, 0], ITris)
    return hks 

#daitailenの関数内で用いる関数平面との交点とその距離を求める関数
def crosspoint(x0,x1,x2,n):
    v = x2-x1
    t = np.abs(np.dot((x0-x1),n)/np.dot(v,n))
#    print(x0-x1,n,np.dot((x0-x1),n))
    point = t*v+x1
    return point,np.linalg.norm(point-x1,ord=2)

#3dオブジェクトに色付けver3
def getcoloroff(inputfile, output, t=10, save = True, neigvecs=200, option = True, circle = True ):
    (VPos, VColors, ITris) = loadOffFile(inputfile)
    vnum = len(VPos)
    neigvecs = min(VPos.shape[0], neigvecs)
    hks = getHKS(VPos, ITris, neigvecs, np.array([t]))
    ris = ITris.flatten()
    if option:
#        lst = [0,13,49,100]
        lst = [0,8,20,30,80,100]
#これがいい感じの位置
#        lst = [0,3,8,38,70,80,100]
#        lst = [0,3,8,38,70,80,100]
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


#3dオブジェクトに色付けver4
#大腿骨の長さを求める
def daitai(inputfile, output, t=10, save = True, neigvecs=200, option = True, circle = True ):
    (VPos, VColors, ITris) = loadOffFile(inputfile)
    vnum = len(VPos)
    neigvecs = min(VPos.shape[0], neigvecs)
    hks = getHKS(VPos, ITris, neigvecs, np.array([t]))
    ris = ITris.flatten()
    if option:
#20が股下の値
        lst = [0,13,30,49,100]
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
    return hks, linevpos, indexpos#, line


#上手い位置を探してその部分の断面積を求めるプログラム
#kmeansのところを本当はxmeansを用いたほうがいい
#plot == Trueのときは関数の外側にしたの一行を付け加える
#fig = plt.figure()#figsize=(10,10))
def solve_a_l(num, array=None, pltnum=None,plot=False):
    #array = [2,3]
    #pltnum = from 1 to array[0]*array[1]
    #num = 0
    time = 1000
    file = 'data/{:03d}.obj'.format(num)
    outputf = 'result/{:03d}'.format(num)
    arealength = np.zeros((17,2))
#手順1(境目の点群を抽出)
    (vpos, clolors, itris) = loadOffFile(file)
    (hks, linevpos, indexpos) = getcoloroff(file,outputf,time,False)
#手順2（境目の点群を各部位にわける）
    kmarray = linevpos
    #idx0だけ異なる手法でクラスタリングを行う（重回帰分析をしてその平面より上かどうかで分ける）
    model_lr = LinearRegression()
    target = (kmarray[indexpos[0]:indexpos[1]])
    (s1,s2,t) = (-target[:,2],target[:,0],target[:,1])
    s = np.transpose(np.vstack((s1,s2)))
    model_lr.fit(s, t)
    u =  model_lr.coef_[0]*s1+model_lr.coef_[1]*s2+model_lr.intercept_-t
    idx0 = np.where(u>0,1,0)
#    if np.mean(linevpos[idx==0,1])>0np.mean(linevpos[idx==1,1]):
    #kmeans++でidx1以降をわける
    kmeans2 = KMeans(n_clusters=2, init='k-means++')#, n_init = 10, max_iter = 10)
    kmeans4 = KMeans(n_clusters=4, init='k-means++')#, n_init = 10, max_iter = 10)
    kmeans5 = KMeans(n_clusters=5, init='k-means++')#, n_init = 10, max_iter = 10)
    idx1 = kmeans2.fit_predict(kmarray[indexpos[1]:indexpos[2]])+2
    idx2 = kmeans5.fit_predict(kmarray[indexpos[2]:indexpos[3]])+2+2
    idx3 = kmeans4.fit_predict(kmarray[indexpos[3]:indexpos[4]])+2+2+5
    idx4 = kmeans4.fit_predict(kmarray[indexpos[4]:indexpos[5]])+2+2+5+4
    idx = np.hstack((idx0,idx1,idx2,idx3,idx4))
####plot####
    if plot:
#        ax = fig.add_subplot(array[0],array[1],pltnum,plojection='3d')
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
    d2 = linevpos
    align = np.zeros((17,3))
    for i in range(17):
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
            comp = pca.components_
            vec = np.cross(comp[0],comp[1])
            vec = vec/vec[2]
            mesh_x1 = np.arange(x1.min(), x1.max(), (x1.max()-x1.min())/20)
            mesh_x2 = np.arange(x2.min(), x2.max(), (x2.max()-x2.min())/20)
            mesh_x1, mesh_x2 = np.meshgrid(mesh_x1, mesh_x2)
            (x10,x20,y0) = np.mean(aim,axis=0)
            mesh_y = -vec[0] * (mesh_x1-x10) + -vec[1] * (mesh_x2-x20) + y0
            ax.plot_wireframe(mesh_x1, mesh_x2, mesh_y,linewidth=0.3)
            exec("ax.plot(aim[:,0],aim[:,1],aim[:,2],marker='.',alpha = 0.3,linestyle='None',label = '%02d')"%(i))
    if plot:
        ax.legend()
####plot####
#    print('area length\n',arealength)
    return arealength


#ver1.2（正規化のために首だけを取り出すコード）
#上手い位置を探してその部分の断面積を求めるプログラム
#kmeansのところを本当はxmeansを用いたほうがいい
#plot == Trueのときは関数の外側にしたの一行を付け加える
#fig = plt.figure()#figsize=(10,10))
def solve_neck(num, array=None, pltnum=None,plot=False):
    #array = [2,3]
    #pltnum = from 1 to array[0]*array[1]
    #num = 0
    time = 1000
    file = 'data/{:03d}.obj'.format(num)
    outputf = 'result/{:03d}'.format(num)
    arealength = np.zeros((17,2))
#手順1(境目の点群を抽出)
    (vpos, clolors, itris) = loadOffFile(file)
    (hks, linevpos, indexpos) = getcoloroff(file,outputf,time,False)
#手順2（境目の点群を各部位にわける）
    kmarray = linevpos
    #idx0だけ異なる手法でクラスタリングを行う（重回帰分析をしてその平面より上かどうかで分ける）
    model_lr = LinearRegression()
    target = (kmarray[indexpos[0]:indexpos[1]])
    (s1,s2,t) = (-target[:,2],target[:,0],target[:,1])
    s = np.transpose(np.vstack((s1,s2)))
    model_lr.fit(s, t)
    u =  model_lr.coef_[0]*s1+model_lr.coef_[1]*s2+model_lr.intercept_-t
    idx0 = np.where(u>0,1,0)
    #kmeans++でidx1以降をわける
    kmeans2 = KMeans(n_clusters=2, init='k-means++')#, n_init = 10, max_iter = 10)
    kmeans4 = KMeans(n_clusters=4, init='k-means++')#, n_init = 10, max_iter = 10)
    kmeans5 = KMeans(n_clusters=5, init='k-means++')#, n_init = 10, max_iter = 10)
    idx1 = kmeans2.fit_predict(kmarray[indexpos[1]:indexpos[2]])+2
    idx2 = kmeans5.fit_predict(kmarray[indexpos[2]:indexpos[3]])+2+2
    idx3 = kmeans4.fit_predict(kmarray[indexpos[3]:indexpos[4]])+2+2+5
    idx4 = kmeans4.fit_predict(kmarray[indexpos[4]:indexpos[5]])+2+2+5+4
    idx = np.hstack((idx0,idx1,idx2,idx3,idx4))
#手順3（主成分分析によりクラスタリングしたものの面積と周長を求める）
    d2 = linevpos
    align = np.zeros((17,3))
    comp = np.zeros((17,2,3))
    for i in range(17):
        (x1,x2,y) = (-d2[idx==i,2],d2[idx==i,0],d2[idx==i,1])
        aim = np.transpose(np.vstack((x1,x2,y)))
        #######idx並び替え#######
        #alignは各境目の集合の平均値(x,y,z)
        align[i] = np.mean(aim,axis=0)
        #######idx並び替え#######
        pca = PCA(n_components=2)
        pca.fit(aim)
        result = pca.transform(aim)
        comp[i] = pca.components_
        points = result[:,:2]
        hull = ConvexHull(points)
        arealength[i] = (hull.volume,hull.area)

#手順4首のインデックスを取得
#######idx並び替え#######
#    base = np.zeros(17,dtype='int32')
    neck = align[4:9]
    if num%10 in [0,1,2,4,8,9]:
        necknum = np.argmax(neck[:,2])+4
    elif num%10 in[3,5,6,7]:
        neckxy = np.abs(neck[:,1])+np.abs(neck[:,0])
        neckxy = np.where(neck[:,2]>0,neckxy,100)
        necknum = np.argmin(neckxy)+4
#######idx並び替え#######
####plot####
    if plot:
#        ax = fig.add_subplot(array[0],array[1],pltnum,plojection='3d')
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
        base = np.arange(17,dtype='int32')
        for h,i in enumerate(base):
            (x1,x2,y) = (-d2[idx==i,2],d2[idx==i,0],d2[idx==i,1])
            aim = np.transpose(np.vstack((x1,x2,y)))
            vec = np.cross(comp[i,0,:],comp[i,1,:])
            vec = vec/vec[2]
            mesh_x1 = np.arange(x1.min(), x1.max(), (x1.max()-x1.min())/20)
            mesh_x2 = np.arange(x2.min(), x2.max(), (x2.max()-x2.min())/20)
            mesh_x1, mesh_x2 = np.meshgrid(mesh_x1, mesh_x2)
            (x10,x20,y0) = np.mean(aim,axis=0)
            mesh_y = -vec[0] * (mesh_x1-x10) + -vec[1] * (mesh_x2-x20) + y0
            ax.plot_wireframe(mesh_x1, mesh_x2, mesh_y,linewidth=0.3)
            exec("ax.plot(aim[:,0],aim[:,1],aim[:,2],marker='.',linestyle='None',label = '%02d')"%(h))
        ax.legend()
####plot####
#    arealength = arealength[base]

        print('necknum',necknum)
        print(arealength)
    temp = arealength[4].copy()
    arealength[4] = arealength[necknum].copy()
    arealength[necknum] = temp
    return arealength

def make_lst():
    num = 20
    num2 = 10
    lst = np.zeros(num*num2,dtype='int32')#ここを変更1
    count = 0
    for i in range(num):#ここを変更1
        for j in range(num2):
            lst[count*num2+j] = int(i*10+j)
        count += 1
    lst = lst[np.where((lst != 11)&(lst != 62)&(lst != 71)&(lst != 102)&(lst != 174)&(lst != 184)&(lst != 194)&(lst != 9)&(lst != 16)&(lst != 28)&(lst != 67)&(lst != 75)&(lst != 77)&(lst != 126)&(lst != 137)&(lst != 156)&(lst != 195)&(lst != 199))]
    return lst

#ver1.3（正規化のために首も取り出しつつ，体積を統一する）
#上手い位置を探してその部分の断面積を求めるプログラム
#kmeansのところを本当はxmeansを用いたほうがいい
#plot == Trueのときは関数の外側にしたの一行を付け加える
#fig = plt.figure()#figsize=(10,10))
#大腿骨の時と体積の時使い分け
#def solve_vol(num, array=None, pltnum=None,plot=False):
def solve_daitai(num, array=None, pltnum=None,plot=False):
    #array = [2,3]
    #pltnum = from 1 to array[0]*array[1]
    #num = 0
    time = 1000
    file = 'data/{:03d}.obj'.format(num)
    outputf = 'result/{:03d}'.format(num)
    arealength = np.zeros((17,2))
#手順1(境目の点群を抽出)
    (vpos, clolors, itris) = loadOffFile(file)
    (hks, linevpos, indexpos) = getcoloroff(file,outputf,time,False)
##############################
##手順1.5(体積が一定になるようにするsolve_volのとき）
#    vol = np.loadtxt('result/vol/meshvol{:03d}.csv'.format(num),delimiter=',')
#    defvol = 200000
#    mugni = (defvol/vol)**(1/3)
##############################
#############################
#手順1.5（大腿骨が一定になるようにするsolve_daitaiのとき）
    dlen = np.loadtxt('result/daitai.csv',delimiter=',')
    deflen = 1
    lst = make_lst()
    lll = np.arange(len(lst))
    mugni = deflen/dlen[lll[lst==num]]
#############################
#ここは共通
    vpos = mugni*vpos
    linevpos = mugni*linevpos
#############################
#手順2（境目の点群を各部位にわける）
    kmarray = linevpos
    #idx0だけ異なる手法でクラスタリングを行う（重回帰分析をしてその平面より上かどうかで分ける）
    model_lr = LinearRegression()
    target = (kmarray[indexpos[0]:indexpos[1]])
    (s1,s2,t) = (-target[:,2],target[:,0],target[:,1])
    s = np.transpose(np.vstack((s1,s2)))
    model_lr.fit(s, t)
    u =  model_lr.coef_[0]*s1+model_lr.coef_[1]*s2+model_lr.intercept_-t
    idx0 = np.where(u>0,1,0)
    #kmeans++でidx1以降をわける
    kmeans2 = KMeans(n_clusters=2, init='k-means++')#, n_init = 10, max_iter = 10)
    kmeans4 = KMeans(n_clusters=4, init='k-means++')#, n_init = 10, max_iter = 10)
    kmeans5 = KMeans(n_clusters=5, init='k-means++')#, n_init = 10, max_iter = 10)
    idx1 = kmeans2.fit_predict(kmarray[indexpos[1]:indexpos[2]])+2
    idx2 = kmeans5.fit_predict(kmarray[indexpos[2]:indexpos[3]])+2+2
    idx3 = kmeans4.fit_predict(kmarray[indexpos[3]:indexpos[4]])+2+2+5
    idx4 = kmeans4.fit_predict(kmarray[indexpos[4]:indexpos[5]])+2+2+5+4
    idx = np.hstack((idx0,idx1,idx2,idx3,idx4))
#手順3（主成分分析によりクラスタリングしたものの面積と周長を求める）
    d2 = linevpos
    align = np.zeros((17,3))
    comp = np.zeros((17,2,3))
    for i in range(17):
        (x1,x2,y) = (-d2[idx==i,2],d2[idx==i,0],d2[idx==i,1])
        aim = np.transpose(np.vstack((x1,x2,y)))
        #######idx並び替え#######
        #alignは各境目の集合の平均値(x,y,z)
        align[i] = np.mean(aim,axis=0)
        #######idx並び替え#######
        pca = PCA(n_components=2)
        pca.fit(aim)
        result = pca.transform(aim)
        comp[i] = pca.components_
        points = result[:,:2]
        hull = ConvexHull(points)
        arealength[i] = (hull.volume,hull.area)

#手順4首のインデックスを取得
#######idx並び替え#######
#    base = np.zeros(17,dtype='int32')
    neck = align[4:9]
    if num%10 in [0,1,2,4,8,9]:
        necknum = np.argmax(neck[:,2])+4
    elif num%10 in[3,5,6,7]:
        neckxy = np.abs(neck[:,1])+np.abs(neck[:,0])
        neckxy = np.where(neck[:,2]>0,neckxy,100)
        necknum = np.argmin(neckxy)+4
#######idx並び替え#######
####plot####
    if plot:
#        ax = fig.add_subplot(array[0],array[1],pltnum,plojection='3d')
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
        base = np.arange(17,dtype='int32')
        for h,i in enumerate(base):
            (x1,x2,y) = (-d2[idx==i,2],d2[idx==i,0],d2[idx==i,1])
            aim = np.transpose(np.vstack((x1,x2,y)))
            vec = np.cross(comp[i,0,:],comp[i,1,:])
            vec = vec/vec[2]
            mesh_x1 = np.arange(x1.min(), x1.max(), (x1.max()-x1.min())/20)
            mesh_x2 = np.arange(x2.min(), x2.max(), (x2.max()-x2.min())/20)
            mesh_x1, mesh_x2 = np.meshgrid(mesh_x1, mesh_x2)
            (x10,x20,y0) = np.mean(aim,axis=0)
            mesh_y = -vec[0] * (mesh_x1-x10) + -vec[1] * (mesh_x2-x20) + y0
            ax.plot_wireframe(mesh_x1, mesh_x2, mesh_y,linewidth=0.3)
            exec("ax.plot(aim[:,0],aim[:,1],aim[:,2],marker='.',linestyle='None',label = '%02d')"%(h))
        ax.legend()
####plot####
#    arealength = arealength[base]

        print('necknum',necknum)
        print(arealength)
    temp = arealength[4].copy()
    arealength[4] = arealength[necknum].copy()
    arealength[necknum] = temp
    return arealength

#大腿骨の長さを求めるプログラムver1
#上手い位置を探してその部分の断面積を求めるプログラム
#kmeansのところを本当はxmeansを用いたほうがいい
#plot == Trueのときは関数の外側にしたの一行を付け加える
#fig = plt.figure()#figsize=(10,10))
def daitailen(num, array=None, pltnum=None,plot=False):
    #array = [2,3]
    #pltnum = from 1 to array[0]*array[1]
    #num = 0
    time = 1000
    file = 'data/{:03d}.obj'.format(num)
    outputf = 'result/{:03d}'.format(num)
#手順1(境目の点群を抽出)
    (vpos, clolors, itris) = loadOffFile(file)
    (hks, linevpos, indexpos) = daitai(file,outputf,time,False)
#手順2（境目の点群を各部位にわける）
    kmarray = linevpos
    #kmeans++でidx0以降をわける
    kmeans2 = KMeans(n_clusters=2, init='k-means++')#, n_init = 10, max_iter = 10)
    kmeans5 = KMeans(n_clusters=5, init='k-means++')#, n_init = 10, max_iter = 10)
    idx0 = kmeans2.fit_predict(kmarray[indexpos[0]:indexpos[1]])
    idx1 = kmeans5.fit_predict(kmarray[indexpos[1]:indexpos[2]])+2
    idx2 = kmeans5.fit_predict(kmarray[indexpos[2]:indexpos[3]])+2+5
    idx = np.hstack((idx0,idx1,idx2))
####plot####
    if plot:
#        ax = fig.add_subplot(array[0],array[1],pltnum,plojection='3d')
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
#5つのインデックスを入れるブロック
    numbrock = np.zeros(5,dtype='int32')
    bcount = 0

#手順3（主成分分析によりクラスタリングしたものの面積と周長を求める）
    d2 = linevpos
    align = np.zeros((12,3))
        #######idx並び替えのためにalign準備#######
    for i in range(12):
        (x1,x2,y) = (-d2[idx==i,2],d2[idx==i,0],d2[idx==i,1])
        aim = np.transpose(np.vstack((x1,x2,y)))
        #alignは各境目の集合の平均値(x,y,z)
        align[i] = np.mean(aim,axis=0)
        #######idx並び替えのためにalign準備#######

    for i in range(12):
        (x1,x2,y) = (-d2[idx==i,2],d2[idx==i,0],d2[idx==i,1])
        aim = np.transpose(np.vstack((x1,x2,y)))
        pca = PCA(n_components=2)
        pca.fit(aim)
        result = pca.transform(aim)
#仕分け##################
        if i<2:
            if i == np.argmax(align[:2,2]):
                continue
                print('continue')
            else:
                numbrock[bcount] = i
                bcount+=1
                comp = pca.components_
        elif i<7:
            if i-2 not in np.argsort(align[2:7,2])[:2]:
                continue
                print('continue')
            else:
                numbrock[bcount] = i
                bcount+=1
        else:
            if i-7 not in np.argsort(align[7:,2])[:2]:
                continue
                print('continue')
            else:
                numbrock[bcount] = i
                bcount+=1
#仕分け##################
####plot####
        if plot:
#平面プロット
            if i<2:
                comp = pca.components_
                vec = np.cross(comp[0],comp[1])
                vec = vec/vec[2]
                mesh_x1 = np.arange(x1.min(), x1.max(), (x1.max()-x1.min())/20)
                mesh_x2 = np.arange(x2.min(), x2.max(), (x2.max()-x2.min())/20)
                mesh_x1, mesh_x2 = np.meshgrid(mesh_x1, mesh_x2)
                (x10,x20,y0) = np.mean(aim,axis=0)
                mesh_y = -vec[0] * (mesh_x1-x10) + -vec[1] * (mesh_x2-x20) + y0
                ax.plot_wireframe(mesh_x1, mesh_x2, mesh_y,linewidth=0.3)
            exec("ax.plot(aim[:,0],aim[:,1],aim[:,2],marker='.',alpha = 0.3,linestyle='None',label = '%02d')"%(i))
            ax.scatter(align[i,0],align[i,1],align[i,2],marker='o')
####plot####
##########solve daitai len############
    i = numbrock[0]
    nvec = np.cross(comp[0],comp[1])
    point = np.zeros((4,3))
    dlen = np.zeros(4)
    (point[0],dlen[0]) = crosspoint(align[i],align[numbrock[3]],align[numbrock[1]],nvec)
    (point[1],dlen[1]) = crosspoint(align[i],align[numbrock[4]],align[numbrock[1]],nvec)
    (point[2],dlen[2]) = crosspoint(align[i],align[numbrock[3]],align[numbrock[2]],nvec)
    (point[3],dlen[3]) = crosspoint(align[i],align[numbrock[4]],align[numbrock[2]],nvec)
#    lsindex = np.argsort(dlen)[:2]
    lsindex = np.argsort(dlen)[:2]
    if plot:
        ax.plot(point[lsindex,0],point[lsindex,1],point[lsindex,2],marker='o',color='black',linestyle='None')
        ax.legend()
##########solve daitai len############
    return np.mean(dlen[lsindex])



##ver2（直立している時に，うまく番号を振り分ける，逆に直立していない3番などではうまくいかない)
##上手い位置を探してその部分の断面積を求めるプログラム
##kmeansのところを本当はxmeansを用いたほうがいい
##plot == Trueのときは関数の外側にしたの一行を付け加える
##fig = plt.figure()#figsize=(10,10))
#def solve_a_l(num, array=None, pltnum=None,plot=False):
#    #array = [2,3]
#    #pltnum = from 1 to array[0]*array[1]
#    #num = 0
#    time = 1000
#    file = 'data/{:03d}.obj'.format(num)
#    outputf = 'result/{:03d}'.format(num)
#    arealength = np.zeros((17,2))
##手順1(境目の点群を抽出)
#    (vpos, clolors, itris) = loadOffFile(file)
#    (hks, linevpos, indexpos) = getcoloroff(file,outputf,time,False)
##手順2（境目の点群を各部位にわける）
#    kmarray = linevpos
#    #idx0だけ異なる手法でクラスタリングを行う（重回帰分析をしてその平面より上かどうかで分ける）
#    model_lr = LinearRegression()
#    target = (kmarray[indexpos[0]:indexpos[1]])
#    (s1,s2,t) = (-target[:,2],target[:,0],target[:,1])
#    s = np.transpose(np.vstack((s1,s2)))
#    model_lr.fit(s, t)
#    u =  model_lr.coef_[0]*s1+model_lr.coef_[1]*s2+model_lr.intercept_-t
#    idx0 = np.where(u>0,1,0)
#    #kmeans++でidx1以降をわける
#    kmeans2 = KMeans(n_clusters=2, init='k-means++')#, n_init = 10, max_iter = 10)
#    kmeans4 = KMeans(n_clusters=4, init='k-means++')#, n_init = 10, max_iter = 10)
#    kmeans5 = KMeans(n_clusters=5, init='k-means++')#, n_init = 10, max_iter = 10)
#    idx1 = kmeans2.fit_predict(kmarray[indexpos[1]:indexpos[2]])+2
#    idx2 = kmeans5.fit_predict(kmarray[indexpos[2]:indexpos[3]])+2+2
#    idx3 = kmeans4.fit_predict(kmarray[indexpos[3]:indexpos[4]])+2+2+5
#    idx4 = kmeans4.fit_predict(kmarray[indexpos[4]:indexpos[5]])+2+2+5+4
#    idx = np.hstack((idx0,idx1,idx2,idx3,idx4))
##手順3（主成分分析によりクラスタリングしたものの面積と周長を求める）
#    d2 = linevpos
#    align = np.zeros((17,3))
#    comp = np.zeros((17,2,3))
#    for i in range(17):
#        (x1,x2,y) = (-d2[idx==i,2],d2[idx==i,0],d2[idx==i,1])
#        aim = np.transpose(np.vstack((x1,x2,y)))
#        #######idx並び替え#######
#        align[i] = np.mean(aim,axis=0)
#        #######idx並び替え#######
#        pca = PCA(n_components=2)
#        pca.fit(aim)
#        result = pca.transform(aim)
#        comp[i] = pca.components_
#        points = result[:,:2]
#        hull = ConvexHull(points)
#        arealength[i] = (hull.volume,hull.area)
#        
##手順4面積と周長を場所順に並び替えるbase配列の作成
########idx並び替え#######
#    base = np.zeros(17,dtype='int32')
#    sortz = np.argsort(align[:,2])
#    sorty = np.argsort(align[:,1])
#    base[:4] = sortz[sortz<4]
#    nec = sortz[-1]
#    base[4] = nec
#    sorty = np.reshape(sorty[(sorty>=4)*(sorty!=nec)],(2,6))
#    temp = align[:,2].copy()
#    temp[sorty[0]] = -5
#    temp = align[:,2]-temp
#    base[5:11] = np.argsort(temp)[-6:]
#    temp = align[:,2].copy()
#    temp[sorty[1]] = -5
#    temp = align[:,2]-temp
#    base[11:] = np.argsort(temp)[-6:]
########idx並び替え#######
#####plot####
#    if plot:
##        ax = fig.add_subplot(array[0],array[1],pltnum,plojection='3d')
#        fig = plt.figure()
#        ax = Axes3D(fig)
#        ax.set_xlim(-0.75,0.75)
#        ax.set_ylim(-0.75,0.75)
#        ax.set_zlim(-0.75,0.75)
#        ax.set_xlabel("X-axis")
#        ax.set_ylabel("Y-axis")
#        ax.set_zlabel("Z-axis")
#        d1 = vpos
#        ax.plot(-d1[:,2],d1[:,0],d1[:,1],",")
#        for h,i in enumerate(base):
#            (x1,x2,y) = (-d2[idx==i,2],d2[idx==i,0],d2[idx==i,1])
#            aim = np.transpose(np.vstack((x1,x2,y)))
#            vec = np.cross(comp[i,0,:],comp[i,1,:])
#            vec = vec/vec[2]
#            mesh_x1 = np.arange(x1.min(), x1.max(), (x1.max()-x1.min())/20)
#            mesh_x2 = np.arange(x2.min(), x2.max(), (x2.max()-x2.min())/20)
#            mesh_x1, mesh_x2 = np.meshgrid(mesh_x1, mesh_x2)
#            (x10,x20,y0) = np.mean(aim,axis=0)
#            mesh_y = -vec[0] * (mesh_x1-x10) + -vec[1] * (mesh_x2-x20) + y0
#            ax.plot_wireframe(mesh_x1, mesh_x2, mesh_y,linewidth=0.3)
#            exec("ax.plot(aim[:,0],aim[:,1],aim[:,2],marker='.',linestyle='None',label = '%02d')"%(h))
#        ax.legend()
#####plot####
#    arealength = arealength[base]
##    print('area length\n',arealength)
#    return arealength
