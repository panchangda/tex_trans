import open3d as o3d
import cv2
import numpy as np
import math

'''
验证opencv UV坐标取法:
u, v正常取
访问image时将u, v对换
'''
dataFile = "./data"
source_mesh = o3d.io.read_triangle_mesh(dataFile + "/source.obj")
target_mesh = o3d.io.read_triangle_mesh(dataFile + "/target.obj")
result_mesh = o3d.io.read_triangle_mesh(dataFile + "/result.obj")

source_image = cv2.imread(dataFile + "/source.png")
target_image = cv2.imread(dataFile + "/target.png")

source_image = cv2.pyrDown(source_image)
source_image = cv2.pyrDown(source_image)
source_image = cv2.pyrDown(source_image)
source_image = cv2.pyrDown(source_image)

source_triangles = np.asarray(source_mesh.triangles)
source_uvs = source_mesh.triangle_uvs
h,w = source_image.shape[:2]
for i in range(source_triangles.shape[0]):
    # uv正常求
    # 访问u,v时反一下
    for index in range(3):
        u = source_uvs[i*3+index][0] * w
        v = h - source_uvs[i*3+index][1] * h
        source_image[int(v),int(u)].fill(0)
        
cv2.imwrite("test.png",source_image)


target_triangles = np.asarray(target_mesh.triangles)
target_uvs = target_mesh.triangle_uvs
h,w = target_image.shape[:2]
for i in range(target_triangles.shape[0]):
    # uv正常求
    # 访问u,v时反一下
    for index in range(3):
        u = target_uvs[i*3+index][0] * w
        v = h - target_uvs[i*3+index][1] * h
        target_image[int(v),int(u)].fill(0)

cv2.imwrite("test.png",target_image)
