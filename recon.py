import math
from math import pi
import time
import cv2
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

radius = 0.5
eps = 1e-2
zNear = math.sqrt(3) - radius - eps
zFar = 1 + zNear + eps


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def ModelViewfromCam(campos):
    f = -campos
    up = np.asarray([0, 1, 0])

    f = normalize(f)
    s = normalize(np.cross(f, up))
    u = normalize(np.cross(s, f))

    R = np.vstack((s, u, -f))
    t = np.asarray([np.dot(s, campos), np.dot(u, campos), np.dot(f, campos)])
    transform = np.eye(4)
    transform[:3,:3] = R
    transform[:3, 3] = t
    return transform

def mmap2point(img, campos):
    ModelView = ModelViewfromCam(campos)
    width, height = img.shape
    f = 0.5 * width / np.tan(0.5 * 43.0 / 180 * pi)

    img = np.flip(img, 0)
    img = zNear + img

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    cu, cv = width // 2, height // 2
    i = xx - cu
    j = yy - cv
    ratio = img / f
    x = (i * ratio).flatten()
    y = (j * ratio).flatten()
    z = (-img).flatten()
    
    points = np.vstack((x, y, z, np.ones_like(z))).T
    points = points[np.where(points[:, 2] > -zFar + eps * 2)]
    points = np.linalg.inv(ModelView).dot(points.T)
    return points.T[:, :3]

def pcd2map(xyz, height, width):
    cx = width // 2
    cy = height // 2
    fx = 0.5 * height / np.tan(0.5 * 43.0 / 180 * pi)
    fy = 0.5 * width / np.tan(0.5 * 43.0 / 180 * pi)
    depth = np.zeros([height, width], dtype=np.uint8)

    for index in range(xyz.shape[0]):
        d = -xyz[index, 2]
        px = int(round(xyz[index, 1] * fy / d + cy))
        py = int(round(xyz[index, 0] * fx / d + cx))
        d = int(round((zFar - d) * 255))

        if d < 0:
            d = 0
        if d > 255:
            d = 255
        if d > depth[px, py]:
            depth[px, py] = d
    
    return np.flip(depth, axis=0)

def vis_pc(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    campos = np.array([-1,-1,-1])
    img = cv2.imread("build/complete_cam_0.png", 0)
    img = 1 - img / 255
    print(img.min(), " ", img.max())
    temp = img[img < 1]
    print(temp.min(), " ", temp.max())
    # img = (img - img.min()) / (img.max() - img.min())

    points = mmap2point(img, campos)
    print(points.shape)
    vis_pc(points)
    points = np.c_[points, np.ones((points.shape[0], 1), dtype=points.dtype)]

    campos1 = np.array([1,1,1])
    T = ModelViewfromCam(campos1)
    points1 = T.dot(points.T).T

    start = time.time()
    img1 = pcd2map(points1[:,:3], 256, 256)
    end = time.time()
    print(end - start)
    
    img2 = cv2.imread("build/complete_cam_6.png", 0)

    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(img1)
    plt.subplot(1,3,3)
    plt.imshow(img2)
    plt.show()
    