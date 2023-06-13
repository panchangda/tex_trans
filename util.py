import math

def bilinear_interp(x,y,image):
    x0 = math.floor(x)
    x1 = x0 + 1
    y0 = math.floor(y)
    y1 = y0 + 1
    ca = image[x0,y0] 
    cb = image[x0,y1]
    cc = image[x1,y0]
    cd = image[x1,y1]
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    return wa*ca + wb*cb + wc*cc + wd*cd     

# def bilinear_interp(x,y,image):

#     return image[math.floor(x), math.floor(y)]
