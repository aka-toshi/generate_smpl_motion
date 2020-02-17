# generate the SMPL motions using vmocap

## make SMPL model from image
code from [HMR](https://github.com/akanazawa/hmr)   
imagefolder is in `./exp/imagedata/example.png`   
run with **python2**(read [HMR](https://github.com/akanazawa/hmr) settings)
```
$.
$make 0=example
```
result SMPL data is in `./exp/objdata/example.obj`

## make SMPL motions using vmocap data(.trc)
vmocapdata (.trc) is in `./exp/trc/exapmle.trc`   
SMPL data is in `./exp/objdata/example.obj`
```
$exp
{yourblenderpass} --python makeanim.py -- example.obj -- {start frame number} -- {frame range number}
```
example
```
~/../../Applications/Blender.app/Contents/MacOS/Blender --python makeanim.py -- model1.obj -- 600 -- 60
```





<!--
```
$conda activate py27
./exp/imagedata/~~.png
$make 0=~~
$cd exp
./objdata/~~~.obj
$make 0=~~~
```
結果として，./poseobjにobjファイルが入る
-->
