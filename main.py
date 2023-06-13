import argparse
from audioop import avg
from calendar import c
import copy
from os import ctermid
import os
import time
from typing import get_args
from cv2 import MIXED_CLONE
import open3d as o3d
import cv2
import numpy as np
import math
from color_transfer import color_transfer
from util import bilinear_interp

tic = time.perf_counter()

rootFolder = "/home/pcd/vscodes/tex_trans"
dataFile = "/home/pcd/vscodes/tex_trans/data"
outFile = "/home/pcd/vscodes/tex_trans/blended_uv.png"

source_mesh_file = dataFile + "/source.obj"
source_image_file = dataFile + "/source.png"

target_mesh_file = dataFile + "/MeInGame.obj"
target_image_file = dataFile + "/MeInGame.png"

refined_image_file = dataFile + "/refined.png"

corresFile = "/home/pcd/vscodes/pcd_nricp/result/Corr.txt"

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t','--target_mesh',type=str)
parser.add_argument('-uv','--target_uv',type=str)
parser.add_argument('-o','--output',type=str)
config = parser.parse_args()

if config.target_mesh:
    target_mesh_file = config.target_mesh
    print("specify target_mesh as {}".format(target_mesh_file))
if config.target_uv:
    target_image_file = config.target_uv
    print("specify target_uv as {}".format(target_image_file))
if config.output:
    outFile = config.output
    print("specify outFile as {}".format(outFile))

# read-ins
source_mesh = o3d.io.read_triangle_mesh(source_mesh_file)
target_mesh = o3d.io.read_triangle_mesh(target_mesh_file)
# result_mesh = o3d.io.read_triangle_mesh(dataFile + "/result.obj")

channel3_mask_uv = cv2.imread(dataFile + "/mask.png")
flawed_mask_uv = cv2.imread(dataFile + "/flaw_uv.png")

source_image = cv2.imread(source_image_file)
target_image = cv2.imread(target_image_file)

refined_image = cv2.imread(refined_image_file)

# correspondences between source <-> target
with open(corresFile, "r") as f:
    data = f.readlines()
data = [line.strip('\n') for line in data]

corr_src = [int (line.split(' ')[0]) for line in data]
corr_tgt = [int (line.split(' ')[1]) for line in data]
corr_dict = dict(zip(corr_src,corr_tgt))


source_triangles = np.asarray(source_mesh.triangles)
target_triangles = np.asarray(target_mesh.triangles)

adjacent_triangles = [[]for i in range(len(source_mesh.vertices))]
for i in range(len(source_triangles)):
    vs = source_triangles[i]
    adjacent_triangles[vs[0]].append(i)
    adjacent_triangles[vs[1]].append(i)
    adjacent_triangles[vs[2]].append(i)

# shape of point_uv:
# ( vertices num, vertices uv num * 2: pair[u,v] ) 
source_point_uv = [[]for i in range(len(source_mesh.vertices))]
target_point_uv = [[]for i in range(len(target_mesh.vertices))]

# ( 3 * triangles num, 2: [u, v] )
source_triangle_uv = np.asarray(source_mesh.triangle_uvs)
target_triangle_uv = np.asarray(target_mesh.triangle_uvs)
# print(source_triangle_uv.shape)

for i in range(len(source_triangles)):
    vs = source_triangles[i]
    source_point_uv[vs[0]].append(source_triangle_uv[i*3])
    source_point_uv[vs[1]].append(source_triangle_uv[i*3+1])
    source_point_uv[vs[2]].append(source_triangle_uv[i*3+2])
for i in range(len(target_triangles)):
    vs = target_triangles[i]
    target_point_uv[vs[0]].append(target_triangle_uv[i*3])
    target_point_uv[vs[1]].append(target_triangle_uv[i*3+1])
    target_point_uv[vs[2]].append(target_triangle_uv[i*3+2])

# np.unique() 指定axis = 0 去除重复行 axis = 1 去除重复列
# 不指定axis, 展开同维的所有数组再去重 
source_point_uv = [ np.unique(p, axis = 0) for p in source_point_uv ]
target_point_uv = [ np.unique(p, axis = 0) for p in target_point_uv ]

# print(target_point_uv[0])
# print(point_uv[0][0])
# print(point_uv[0][0][0],point_uv[0][0][1])

# 将原贴图降采到目标大小
# source_image.shape[0] / target_image.shape[0] = 2
for i in range(2):
    source_image = cv2.pyrDown(source_image)
    channel3_mask_uv = cv2.pyrDown(channel3_mask_uv)
for i in range(0):
    target_image = cv2.pyrUp(target_image)


src_h, src_w = source_image.shape[:2]
tgt_h, tgt_w = target_image.shape[:2]
uv_size = src_h - 1

# print(source_image.shape, target_image.shape, mask_uv.shape)

mask_uv = np.zeros((src_h,src_w,1), np.uint8)
# mask_uv_inv = np.zeros((src_h,src_w,1), np.uint8)

mask_uv.fill(0)
# mask_uv_inv.fill(255)

for i in range(channel3_mask_uv.shape[0]):
    for j in range(channel3_mask_uv.shape[1]):
        if channel3_mask_uv[i][j].any() != 0:
            mask_uv[i][j].fill(255)

# flaw_uv = np.zeros((src_h, src_w,1),np.uint8)
# flaw_uv.fill(0)

# for i in range(flawed_mask_uv.shape[0]):
#     for j in range(flawed_mask_uv.shape[1]):
#         if flawed_mask_uv[i][j].any() != 0:
#             flaw_uv[i][j].fill(255)


# cv2.imwrite("flaw_uv.png",flaw_uv)
# cv2.imwrite("masks_uv.png", mask_uv)
# cv2.imwrite("masks_uv_inv.png", mask_uv_inv)

transfered_uv = np.zeros((src_h,src_w,3), np.uint8)
transfered_uv.fill(0)

one_verts_notcorr,two_verts_notcorr,three_verts_notcorr = [],[],[]

# how many triangles are fully corresponded
fully_corresponded_triangles_num = 0
for i in range(source_triangles.shape[0]):
    # 步骤1：查看三角形三个点是否都有对应点
    triangle = source_triangles[i]
    triangle_fully_corresponded = True
    cnt = 0
    for vert in triangle:
        if not vert in corr_dict:
            triangle_fully_corresponded = False
            cnt+=1
    # 问题1：没有完全对应的三角形如何处理
    # 特点：该种三角形个数较少，所有包含boundry点的三角形都是非完全对应的
    if not triangle_fully_corresponded:
        if cnt == 1:
            one_verts_notcorr.append(i)
        if cnt == 2:
            two_verts_notcorr.append(i)
        if cnt == 3:
            three_verts_notcorr.append(i)            
        continue
    fully_corresponded_triangles_num+=1
    corr_triangle = []
    for vert in triangle:
        corr_triangle.append(corr_dict[vert])
    
    # 根据点坐标获取uv
    # 只获取第一个
    corr_uvs_1 = target_point_uv[corr_triangle[0]][0]
    corr_uvs_2 = target_point_uv[corr_triangle[1]][0]
    corr_uvs_3 = target_point_uv[corr_triangle[2]][0]

    # 将uv转换为与opencv坐标一致
    corr_v1 = [uv_size * (1 - corr_uvs_1[1]), corr_uvs_1[0] * uv_size]
    corr_v2 = [uv_size * (1 - corr_uvs_2[1]), corr_uvs_2[0] * uv_size]
    corr_v3 = [uv_size * (1 - corr_uvs_3[1]), corr_uvs_3[0] * uv_size]

    corr_v1 = np.asarray(corr_v1)
    corr_v2 = np.asarray(corr_v2)
    corr_v3 = np.asarray(corr_v3)

    # 问题2：存在有多对一，存在重复（覆盖）上色
    # 多对一也会导致对应出来三个点不一定是三角形，也有可能是一个点、一条边
    # print("{0[0]} {0[1]} {0[2]} corr to {1[0]} {1[1]} {1[2]}".format(triangle,corr_triangle))

    # 步骤2： bounding box + barrycentric coordinates
    # !!!!!!
    # array[start:end] :-> [array[start] ... array[end-1]]
    # u, v正常求 然后将u, v对换得到传入opencv作为坐标
    v = uv_size* source_triangle_uv[i*3:i*3+3,0]
    u = uv_size* (1 - source_triangle_uv[i*3:i*3+3,1])

    minU = min(u)
    minU = int(minU)
    maxU = max(u)
    maxU = math.ceil(maxU)

    minV = min(v)
    minV = int(minV)
    maxV = max(v)
    maxV = math.ceil(maxV)
   
    v1 = np.array([u[0],v[0]])
    v2 = np.array([u[1],v[1]])
    v3 = np.array([u[2],v[2]])

    # 如果点全在mask之外，没有赋值的必要
    # if mask_uv[int(u[0]),int(v[0])]==0 and \
    #      mask_uv[int(u[1]),int(v[1])]==0 and \
    #         mask_uv[int(u[2]),int(v[2])]==0:
    #     continue

    vec_v1v2 = v2 - v1
    vec_v1v3 = v3 - v1
    S_triangle = np.cross(vec_v1v2, vec_v1v3)
    
    cnt = 0
    for u in range(minU, maxU, 1):
        for v in range(minV, maxV, 1):
            vec_tmp = [u,v] - v1
            s = np.cross(vec_tmp, vec_v1v3) / S_triangle
            t = np.cross(vec_v1v2, vec_tmp) / S_triangle

            # 该整数点在三角形内
            if s>=0 and t>=0 and s+t <= 1:
                cnt+=1
                #根据s,t,1-s-t计算对应重心坐标
                cooresponding_barrycentric_coordinates = (1-s-t)* corr_v1 + s* corr_v2 + t* corr_v3
                # 双线性插值
                x,y = cooresponding_barrycentric_coordinates
                transfered_uv[u,v] = bilinear_interp(x,y,target_image)
    

    print("Triangle: {} 's bbox size is {}".format(i,(maxU-minU) * (maxV-minV)))
    print("{} of {} points ({:.2%}) are inside triangle".format(cnt,(maxU-minU) * (maxV-minV),float(cnt)/float((maxU-minU) * (maxV-minV))))

print("{} of {} ({:.2%}) source triangles are corresponded".format(fully_corresponded_triangles_num, len(source_triangles), float(fully_corresponded_triangles_num)/float(len(source_triangles))))

print("{} {} {} are size of unfullycorrs".format(len(one_verts_notcorr),len(two_verts_notcorr),len(three_verts_notcorr)))

# cnts = cv2.findContours(mask_uv, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0]
# x, y, w, h = cv2.boundingRect(cnts[0])
# cX = x + w//2
# cY = y + h//2
# # print(cX, cY)

# color_transfer_mask = cv2.imread(dataFile + "/skin_mask.png",cv2.IMREAD_GRAYSCALE)
# source_image = color_transfer(target_image,source_image, color_transfer_mask)

# masked_result = cv2.seamlessClone(transfered_uv, source_image, mask_uv, (cX,cY), flags = cv2.MIXED_CLONE )
# cv2.imwrite("masked_result.png", masked_result)
# masked_result_inv = cv2.seamlessClone(source_image, transfered_uv, mask_uv, (cX,cY), flags = cv2.MIXED_CLONE )
# cv2.imwrite("masked_result_inv.png", masked_result_inv)
# mask_result_normal = cv2.seamlessClone(transfered_uv, source_image, mask_uv, (cX,cY), flags = cv2.NORMAL_CLONE )
# mask_result_normal_inv = cv2.seamlessClone(source_image, transfered_uv,mask_uv, (cX,cY), flags = cv2.NORMAL_CLONE )
# cv2.imwrite("mask_normal.png",mask_result_normal)
# cv2.imwrite("mask_normal_inv.png",mask_result_normal_inv)

result = copy.deepcopy(transfered_uv)
for i in range(1,8):
    mask = cv2.imread(dataFile + "/flaw_area/mask{}.png".format(i))
    mask_binary = np.zeros((src_h, src_w,1),np.uint8)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x][y].any() != 0:
                mask_binary[x][y].fill(255)

    cnts = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    x, y, w, h = cv2.boundingRect(cnts[0])
    cX = x + w//2
    cY = y + h//2
    print(cX, cY)
    result = cv2.seamlessClone(refined_image, result, mask_binary, (cX,cY), flags = cv2.NORMAL_CLONE)

cv2.imwrite(outFile, result)

# count time
toc = time.perf_counter()
print("Total Time used: {:.2f}s".format(toc - tic))


# transfer color to body uv map
body_component_names=["body.png","arm.png","leg.png"]
color_transfer_mask = cv2.imread(dataFile + "/skin_mask.png",cv2.IMREAD_GRAYSCALE)
outfolder,filename = os.path.split(outFile)
filename = filename.split('.')[0]
for i in range(len(body_component_names)):
    color_transfer_target = cv2.imread(dataFile+ "/" + body_component_names[i])
    color_transfer_result = color_transfer(target_image,color_transfer_target,color_transfer_mask)
    cv2.imwrite(outfolder + "/" + filename + "_" + body_component_names[i],color_transfer_result)


exit()

newly_filled_v = {}

# 用两个已有颜色点插值整个三角形
for i in one_verts_notcorr:
    triangle = source_triangles[i]  
    corr_v = []
    not_corr_v = 0
    for j in range(3):
        if triangle[j] not in corr_dict:
            not_corr_v = j
        else:
            corr_v.append(j)

    corr_triangle = []
    for j in corr_v:
        corr_triangle.append(corr_dict[triangle[j]])

    # 根据点坐标获取uv
    # 只获取第一个
    corr_uvs_1 = target_point_uv[corr_triangle[0]][0]
    corr_uvs_2 = target_point_uv[corr_triangle[1]][0]

    # 将uv转换为与opencv坐标一致
    corr_v1 = [uv_size * (1 - corr_uvs_1[1]), corr_uvs_1[0] * uv_size]
    corr_v2 = [uv_size * (1 - corr_uvs_2[1]), corr_uvs_2[0] * uv_size]

    corr_v1 = np.asarray(corr_v1)
    corr_v2 = np.asarray(corr_v2)

    color_v1 = bilinear_interp(corr_v1[0],corr_v1[1],target_image)
    color_v2 = bilinear_interp(corr_v2[0],corr_v2[1],target_image)
    
    v = uv_size* source_triangle_uv[i*3:i*3+3,0]
    u = uv_size* (1 - source_triangle_uv[i*3:i*3+3,1])

    v1 = np.array([u[corr_v[0]],v[corr_v[0]]])
    v2 = np.array([u[corr_v[1]],v[corr_v[1]]])
    v3 = np.array([u[not_corr_v],v[not_corr_v]])

    minU = min(u)
    minU = int(minU)
    maxU = max(u)
    maxU = math.ceil(maxU)

    minV = min(v)
    minV = int(minV)
    maxV = max(v)
    maxV = math.ceil(maxV)

    vec_v1v2 = v2 - v1
    vec_v1v3 = v3 - v1
    
    S_triangle = np.cross(vec_v1v2, vec_v1v3)
    
    bbox=0
    for u in range(minU, maxU, 1):
        for v in range(minV, maxV, 1):
            vec_tmp = [u,v] - v1
            s = np.cross(vec_tmp, vec_v1v3) / S_triangle
            t = np.cross(vec_v1v2, vec_tmp) / S_triangle
            # 该整数点在三角形内
            if s>=0 and t>=0 and s+t <= 1:
                bbox+=1
                transfered_uv[u,v] = (1-s-t)/(1-t) * color_v1 + (s)/(1-t) * color_v2
                # transfered_uv[u,v] = [0,0,255]
    print("Triangle: {} 's bbox size is {}".format(i,(maxU-minU) * (maxV-minV)))
    print("{} of {} points ({:.2%}) are inside triangle".format(bbox,(maxU-minU) * (maxV-minV),float(bbox)/float((maxU-minU) * (maxV-minV))))
    
    newly_filled_v[triangle[not_corr_v]] = 0.3* color_v1 + 0.3* color_v2

for i in two_verts_notcorr:
    triangle = source_triangles[i]
    triangle = source_triangles[i]   
    used_corr = 3
    newly_corr = []
    not_corr = []
    flag = False
    for j in range(3):
        if triangle[j] in corr_dict:
            used_corr = j
        elif triangle[j] in newly_filled_v:
            newly_corr.append(j)
        else:
            not_corr.append(j)
    
    v = uv_size* source_triangle_uv[i*3:i*3+3,0]
    u = uv_size* (1 - source_triangle_uv[i*3:i*3+3,1])

    Points = 0
    color = [0,0,0]
    for j in adjacent_triangles[triangle[used_corr]]:
        adjacent_triangle = source_triangles[j]
        flag = True
        for vert in adjacent_triangle:
            if vert not in corr_dict:
                flag = False
                break

        if flag == False:
            break

        corr_triangle = []
        for vert in adjacent_triangle:
            corr_triangle.append(corr_dict[vert])
    
        corr_uvs_1 = target_point_uv[corr_triangle[0]][0]
        corr_uvs_2 = target_point_uv[corr_triangle[1]][0]
        corr_uvs_3 = target_point_uv[corr_triangle[2]][0]

        # 将uv转换为与opencv坐标一致
        corr_v1 = [uv_size * (1 - corr_uvs_1[1]), corr_uvs_1[0] * uv_size]
        corr_v2 = [uv_size * (1 - corr_uvs_2[1]), corr_uvs_2[0] * uv_size]
        corr_v3 = [uv_size * (1 - corr_uvs_3[1]), corr_uvs_3[0] * uv_size]

        corr_v1 = np.asarray(corr_v1)
        corr_v2 = np.asarray(corr_v2)
        corr_v3 = np.asarray(corr_v3)

        color_v1 = bilinear_interp(corr_v1[0],corr_v1[1],target_image)
        color_v2 = bilinear_interp(corr_v2[0],corr_v2[1],target_image)
        color_v3 = bilinear_interp(corr_v3[0],corr_v3[1],target_image)

        color = 0.33 * color_v1 + 0.33 * color_v2 + 0.33 * color_v3
    
    if all(color) == False:
        for j in adjacent_triangles[triangle[used_corr]]:
            
            not_corr  = []
            adjacent_triangle = source_triangles[j]
            
            for vert in adjacent_triangle:
                if not vert in corr_dict:
                    not_corr.append(vert)
            if len(not_corr) != 1:
                continue
            
            index = np.where(adjacent_triangle==not_corr[0])
            adjacent_triangle = np.delete(adjacent_triangle,index)

            corr_triangle = []
            for vert in adjacent_triangle:
                corr_triangle.append(corr_dict[vert])
    
            corr_uvs_1 = target_point_uv[corr_triangle[0]][0]
            corr_uvs_2 = target_point_uv[corr_triangle[1]][0]


            # 将uv转换为与opencv坐标一致
            corr_v1 = [uv_size * (1 - corr_uvs_1[1]), corr_uvs_1[0] * uv_size]
            corr_v2 = [uv_size * (1 - corr_uvs_2[1]), corr_uvs_2[0] * uv_size]

            corr_v1 = np.asarray(corr_v1)
            corr_v2 = np.asarray(corr_v2)


            color_v1 = bilinear_interp(corr_v1[0],corr_v1[1],target_image)
            color_v2 = bilinear_interp(corr_v2[0],corr_v2[1],target_image)

            color = 0.5 * color_v1 + 0.5 * color_v2 
    
    if all(color) == False:
        for j in adjacent_triangles[triangle[used_corr]]:
            
            corr_v = 3
            not_corr_cnt = 0
            adjacent_triangle = source_triangles[j]
            
            for vert in adjacent_triangle:
                if not vert in corr_dict:
                    not_corr_cnt+=1
                else:
                    corr_v = vert

            if not_corr_cnt != 2:
                continue
    
            corr_uvs_1 = target_point_uv[corr_dict[corr_v]][0]
            # 将uv转换为与opencv坐标一致
            corr_v1 = [uv_size * (1 - corr_uvs_1[1]), corr_uvs_1[0] * uv_size]
            corr_v1 = np.asarray(corr_v1)
            
            color = bilinear_interp(corr_v1[0],corr_v1[1],target_image)


    minU = min(u)
    minU = int(minU)
    maxU = max(u)
    maxU = math.ceil(maxU)

    minV = min(v)
    minV = int(minV)
    maxV = max(v)
    maxV = math.ceil(maxV)
        
    v1 = np.array([u[0],v[0]])
    v2 = np.array([u[1],v[1]])
    v3 = np.array([u[2],v[2]])

    # 如果点全在mask之外，没有赋值的必要
    # if mask_uv[int(u[0]),int(v[0])]==0 and \
    #      mask_uv[int(u[1]),int(v[1])]==0 and \
    #         mask_uv[int(u[2]),int(v[2])]==0:
    #     continue

    vec_v1v2 = v2 - v1
    vec_v1v3 = v3 - v1
    
    S_triangle = np.cross(vec_v1v2, vec_v1v3)
    cnt = 0
    for u in range(minU, maxU, 1):
        for v in range(minV, maxV, 1):
            vec_tmp = [u,v] - v1
            s = np.cross(vec_tmp, vec_v1v3) / S_triangle
            t = np.cross(vec_v1v2, vec_tmp) / S_triangle
            # 该整数点在三角形内
            if s>=0 and t>=0 and s+t <= 1:
                transfered_uv[u,v] = color

# for i in two_verts_notcorr:
#     triangle = source_triangles[i]   
#     used_corr = 3
#     newly_corr = []
#     not_corr = []
#     flag = False
#     for j in range(3):
#         if triangle[j] in corr_dict:
#             used_corr = j
#         elif triangle[j] in newly_filled_v:
#             newly_corr.append(j)
#         else:
#             not_corr.append(j)

#     corr_uvs_1 = target_point_uv[corr_dict[triangle[used_corr]]][0]
#     corr_v1 = [uv_size * (1 - corr_uvs_1[1]), corr_uvs_1[0] * uv_size]
#     corr_v1 = np.asarray(corr_v1)
#     color_v1 = bilinear_interp(corr_v1[0],corr_v2[0],target_image)

#     v = uv_size* source_triangle_uv[i*3:i*3+3,0]
#     u = uv_size* (1 - source_triangle_uv[i*3:i*3+3,1])

#     # always has 1 used_corr
#     v1 = np.array([u[used_corr],v[used_corr]])

#     # 2 newly_corr 
#     if len(newly_corr) == 2:
#         v2 = np.array([u[newly_corr[0]],v[newly_corr[0]]])
#         v3 = np.array([u[newly_corr[1]],v[newly_corr[1]]])
#         color_v2 = newly_filled_v[triangle[newly_corr[0]]]
#         color_v3 = newly_filled_v[triangle[newly_corr[1]]]
        
#     # 1 newly_corr + 1 not_corr
#     elif len(newly_corr) == 1:
#         v2 = np.array([u[newly_corr[0]],v[newly_corr[0]]])
#         v3 = np.array([u[not_corr[0]],v[not_corr[0]]])

#         color_v2 = newly_filled_v[triangle[newly_corr[0]]]

#         newly_filled_v[triangle[not_corr[0]]] = 0.5* color_v1 + 0.5 * color_v2 
#     # 2 not_corr
#     else:
#         v2 = np.array([u[not_corr[0]],v[not_corr[0]]])
#         v3 = np.array([u[not_corr[1]],v[not_corr[1]]])

#         newly_filled_v[triangle[not_corr[0]]] = color_v1
#         newly_filled_v[triangle[not_corr[1]]] = color_v1


#     minU = min(u)   
#     minU = int(minU)
#     maxU = max(u)
#     maxU = math.ceil(maxU)

#     minV = min(v)
#     minV = int(minV)
#     maxV = max(v)
#     maxV = math.ceil(maxV)

#     vec_v1v2 = v2 - v1
#     vec_v1v3 = v3 - v1
    
#     S_triangle = np.cross(vec_v1v2, vec_v1v3)
    
#     bbox = 0
#     for u in range(minU, maxU, 1):
#         for v in range(minV, maxV, 1):
#             vec_tmp = [u,v] - v1
#             s = np.cross(vec_tmp, vec_v1v3) / S_triangle
#             t = np.cross(vec_v1v2, vec_tmp) / S_triangle
#             # 该整数点在三角形内
#             if s>=0 and t>=0 and s+t <= 1:
#                 bbox+=1
#                 if len(newly_corr) == 2:
#                     transfered_uv[u,v] = (1-s-t) * color_v1 + s * color_v2 + t * color_v3
#                 # elif len(newly_corr) == 1:
#                 #     transfered_uv[u,v] = (1-s-t)/(1-t) * color_v1 + (s)/(1-t) * color_v2 
#                 else:
#                     transfered_uv[u,v] = color_v1
#     print("Triangle: {} 's bbox size is {}".format(i,(maxU-minU) * (maxV-minV)))
#     print("{} of {} points ({:.2%}) are inside triangle".format(bbox,(maxU-minU) * (maxV-minV),float(bbox)/float((maxU-minU) * (maxV-minV))))





for i in three_verts_notcorr:
    triangle = source_triangles[i]
    newly_corr = []
    not_corr = []

    for j in range(3):
        if triangle[j] in newly_filled_v:
            newly_corr.append(j)
        else:
            not_corr.append(j)
            cnt+=1
    
    v = uv_size* source_triangle_uv[i*3:i*3+3,0]
    u = uv_size* (1 - source_triangle_uv[i*3:i*3+3,1])

    minU = min(u)
    minU = int(minU)
    maxU = max(u)
    maxU = math.ceil(maxU)

    minV = min(v)
    minV = int(minV)
    maxV = max(v)
    maxV = math.ceil(maxV)

    # 2 newly_corr + 1 not corr
    if cnt == 1:
        v1 = np.array([u[newly_corr[0]],v[newly_corr[0]]])
        v2 = np.array([u[newly_corr[1]],v[newly_corr[1]]])
        v3 = np.array([u[not_corr[0]],v[not_corr[0]]])

        vec_v1v2 = v2 - v1
        vec_v1v3 = v3 - v1
    
        S_triangle = np.cross(vec_v1v2, vec_v1v3)

        color_v1 = newly_filled_v[triangle[newly_corr[0]]]
        color_v2 = newly_filled_v[triangle[newly_corr[1]]]
        bbox = 0
        for u in range(minU, maxU, 1):
            for v in range(minV, maxV, 1):
                vec_tmp = [u,v] - v1
                s = np.cross(vec_tmp, vec_v1v3) / S_triangle
                t = np.cross(vec_v1v2, vec_tmp) / S_triangle
                # 该整数点在三角形内
                if s>=0 and t>=0 and s+t <= 1:
                    bbox+=1
                    # transfered_uv[u,v] = (1-s-t)/(1-t) * color_v1 + (s)/(1-t) * color_v2
                    transfered_uv[u,v] = [255,0,0]
        print("Triangle: {} 's bbox size is {}".format(i,(maxU-minU) * (maxV-minV)))
        print("{} of {} points ({:.2%}) are inside triangle".format(bbox,(maxU-minU) * (maxV-minV),float(bbox)/float((maxU-minU) * (maxV-minV))))
    
    # 1 newly_corr + 2 not corr
    if cnt == 2:
        v1 = np.array([u[newly_corr[0]],v[newly_corr[0]]])
        v2 = np.array([u[not_corr[1]],v[not_corr[1]]])
        v3 = np.array([u[not_corr[0]],v[not_corr[0]]])

        vec_v1v2 = v2 - v1
        vec_v1v3 = v3 - v1
    
        S_triangle = np.cross(vec_v1v2, vec_v1v3)
        
        color_v1 = newly_filled_v[triangle[newly_corr[0]]]
        bbox = 0
        for u in range(minU, maxU, 1):
            for v in range(minV, maxV, 1):
                vec_tmp = [u,v] - v1
                s = np.cross(vec_tmp, vec_v1v3) / S_triangle
                t = np.cross(vec_v1v2, vec_tmp) / S_triangle
                # 该整数点在三角形内
                if s>=0 and t>=0 and s+t <= 1:
                    bbox+=1
                    # transfered_uv[u,v] = color_v1
                    transfered_uv[u,v] = [255,0,0]
        print("Triangle: {} 's bbox size is {}".format(i,(maxU-minU) * (maxV-minV)))
        print("{} of {} points ({:.2%}) are inside triangle".format(bbox,(maxU-minU) * (maxV-minV),float(bbox)/float((maxU-minU) * (maxV-minV))))
    
    if cnt == 3:
        print("3 of 3 not corr") 






# find mask center by its boundry
# cnts = cv2.findContours(mask_uv, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0]
# x, y, w, h = cv2.boundingRect(cnts[0])
# cX = x + w//2
# cY = y + h//2

# color_change 加强颜色
# color_change = cv2.colorChange(transfered_uv, mask_uv, transfered_uv,1.5,1.5,1.5)

# blended_uv = cv2.seamlessClone(transfered_uv, source_image, mask_uv, (cX,cY), flags = cv2.MIXED_CLONE )
# cv2.imwrite(outFile, blended_uv)
# cv2.imwrite("colorchange.png",color_change)


cv2.imwrite(outFile, transfered_uv)
# count time
toc = time.perf_counter()
print("Total Time used: {:.2f}s".format(toc - tic))

exit()

'''
色差过大改进方案：
1. 把mask中眼睛部分画大一些点
2. 降低color_change的系数
颜色变化过小改进方案：
1. 增大mask
2. 直接全部使用新生成的uv颜色（在mask中）
transfered_uv颜色还是有问题
1. 插值公式
2. 特殊区域特殊处理
'''