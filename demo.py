"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

#from smpl.smpl_webuser.hello_world import makesmpl

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam, filename):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.savefig("exp/figure/"+filename+"-a.png",format = 'png',transparent=True, dpi=800)
    plt.clf()
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.savefig("exp/figure/"+filename+"-b.png",format = 'png',transparent=True, dpi=800)
    plt.clf()
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.savefig("exp/figure/"+filename+"-c.png",format = 'png',transparent=True, dpi=800)

def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

def main(img_path, write=False, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

#    imgpath = '/User/akase/lab/hks/src/finaldefense/image/'+img_path
    imgpath = 'exp/imgdata/'+img_path

    input_img, proc_param, img = preprocess_image(imgpath, json_path)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    if write:
        visualize(img, proc_param, joints[0], verts[0], cams[0], img_path[:-4])

    np.savetxt('exp/theta/theta'+img_path[:-4]+'.csv',theta[0],delimiter=',')
    np.savetxt('exp/joint/joint'+img_path[:-4]+'.csv',joints3d[0],delimiter=',')
    np.savetxt('exp/verts/verts'+img_path[:-4]+'.csv',verts[0],delimiter=',')


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1
    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
    main(config.img_path,False,config.json_path)
