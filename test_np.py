import numpy as np
import cv2
import os
# channel3_mask_uv = cv2.imread("masks_uv.png")
# mask_uv = np.zeros((1024,1024,1), np.uint8)
# mask_uv_inv = np.zeros((1024,1024,1), np.uint8)
# mask_uv.fill(0)
# mask_uv_inv.fill(255)
# for i in range(channel3_mask_uv.shape[0]):
#     for j in range(channel3_mask_uv.shape[1]):
#         if channel3_mask_uv[i][j].any() != 0:
#             mask_uv[i][j].fill(255)
#             mask_uv_inv[i][j].fill(0)
# cnts = cv2.findContours(mask_uv, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# cnts = cnts[0]
# print(cnts[0])
# x, y, w, h = cv2.boundingRect(cnts[0])
# print(x,y,w,h)
# cX = x + w//2
# cY = y + h//2



# t = "C:/asdasd/asdwe/231231/sb.fbx"
# f1,f2 = os.path.split(t)
# print(f1)

# a = b = c = 1
# print(a,b,c)




source = cv2.imread("data/3_uv.png")
mask = cv2.imread("data/skin_mask.png",cv2.IMREAD_GRAYSCALE)

l, a, b = cv2.split(source)

lMean = l.mean()
print(lMean)

lMean, lStd = cv2.meanStdDev(l,mask=mask)
lMean=lMean[0][0]
print(lMean,lStd)

lSum = 0
cnt = 0
for i in range(1024):
    for j in range(1024):
        if mask[i][j].tolist() != 0:
            lSum += source[i][j][0]
            cnt += 1

lMean = lSum / cnt
print(lMean)



