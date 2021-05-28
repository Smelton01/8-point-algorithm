#%% 
import numpy as np
import cv2
from matplotlib import pyplot as plt
# %%
# Corresponding points for close image with different planes
uvMat = [[2652, 2001, 2002, 1890], [2714, 1870, 2270, 1788], [1884, 750, 1254, 
   684], [2660, 2181, 2000, 2055], [1980, 1780, 1370, 1580], [3641, 2373, 3150, 2505], [2113, 2290, 1880, 2088], [3040, 1975, 3146, 1993]]

# Corresponding points for close image in approximately same plane
uvMat = [[3641, 2373, 3150, 2505], [2655, 2522, 2007, 2375], [2883, 2414, 2340, 
   2347], [890, 2307, 1033, 1900], [2113, 2290, 1880, 2088], [3040, 1975, 
   3146, 1993], [2255, 2360,1902, 2170], [2044, 2600, 1465, 2306]]

# Corresponding points for far image with different planes
uvMat = [[1787, 1973, 1536, 1847], [2520, 2290, 1677, 2177], [2830, 2099, 1925, 
   2077], [3410, 2390, 2831, 2430], [2870, 2850, 1810, 2724], [2083, 1030, 1790, 1030], [1770, 1108, 1500, 1072], [3370, 2137, 3070, 2200]]


# %%
def load_imgs():
    close1  = plt.imread("close1.jpg")
    close2 = plt.imread('close2.jpg')
    far1  = plt.imread("far1.jpg")
    far2 = plt.imread('far2.jpg')
    return close1, close2, far1, far2
close1, close2, far1, far2 = load_imgs()

# %%
from functools import reduce
def normalize(points):
    n = len(points)
    pt1, pt2 = (0,0), (0,0)
    img1_pts, img2_pts = [], []
    for a,b,c,d in points:
        img1_pts.append([a,b])
        img2_pts.append([c,d])
        # pt1 = (pt1[0]+a, pt1[1]+b)
        # pt2 = (pt2[0]+c, pt2[1]+d)
    pt1 = reduce(lambda x, y:  (x[0]+y[0], x[1]+y[1]), img1_pts)
    pt2 = reduce(lambda x, y:  (x[0]+y[0], x[1]+y[1]), img2_pts)
    
    pt1 = [val/n for val in pt1]
    pt2 = [val/n for val in pt2]
    print("mean1 ", pt1)
    s1 = 2**0.5/(8*sum([((x-pt1[0])**2 + (y-pt1[1])**2)**0.5 for x,y in img1_pts]))
    s2 = 2**0.5/(8*sum([((x-pt2[0])**2 + (y-pt2[1])**2)**0.5 for x,y in img2_pts]))

    T1 = np.array([[s1, 0, -pt1[0]*s1], [0, s1, -pt1[1]*s1], [0, 0, 1]])

    T2 = np.array([[s2, 0, -pt2[0]*s2], [0, s2, -pt2[1]*s2], [0, 0, 1]])

    return [[T1 @ [a,b,1], T2 @ [c, d, 1]] for a,b,c,d in points], T1, T2
# %%
points, T1, T2 = normalize(uvMat)
print(points[0])
# %%

def calc_F(points):
    A = np.zeros((8,9))
    # calculate A matrix with x aand x' swapped
    uvMat = [[r[0], r[1], l[0], l[1]] for l,r in points]

    for i in range(len(uvMat)):
        A[i][0] = uvMat[i][0]*uvMat[i][2]
        A[i][1] = uvMat[i][1]*uvMat[i][2]
        A[i][2] = uvMat[i][2]
        A[i][3] = uvMat[i][0]*uvMat[i][3]
        A[i][4] = uvMat[i][1]*uvMat[i][3]
        A[i][5] = uvMat[i][3] 
        A[i][6] = uvMat[i][0]
        A[i][7] = uvMat[i][1]
        A[i][8] = 1.0  
  
    transAA = np.matmul(A.transpose(),A)
    w1 , v1 = np.linalg.eig(transAA)
    f_vec = v1[:,8]
    f_hat = np.reshape(f_vec, (3,3))

    s,v,d = np.linalg.svd(f_hat)
    f_hat = s @ np.diag([*v[:2], 0]) @ d.transpose()

    return f_hat
f_hat = calc_F(points)
print(f_hat)
# %%
# Restore to the original coordinates
def restore(f_hat, T1, T2):
    T2_trans = T2.transpose()

    f_mat = T2_trans @ f_hat 
    f_mat = f_mat @ T1
    return f_mat 

f_mat = restore(f_hat, T1, T2)
print(f_mat)

#%%
def plot1(img, f_mat, points):
    w = img.shape[1]
    for *pt, x2, y2 in points:
        a,b,c = np.array([*pt, 1]).transpose() @ f_mat
        p1 = (0,-c/b)
        p2 = (w, -(a*w + c)/b)
        plt.plot(*zip(p1,p2))
        plt.plot(x2, y2, "x")
    plt.imshow(img)

#%%
def plot2(img, f_mat, points):
    w = img.shape[1]
    for x1,y1, *pt in points:
        a,b,c = f_mat @ [*pt, 1]
        p1 = (0,-c/b)
        p2 = (w, -(a*w + c)/b)
        plt.plot(*zip(p1,p2))
        plt.plot(x1,y1, "x")
    plt.imshow(img)

#%%

# Plot the epipolar lines
plot1(far2, f_mat, uvMat) 

#%%
plot2(far1, f_mat, uvMat)

# %%
######################################################
# AtransA = np.matmul(A, transA)
transAA = np.matmul(A.transpose(),A)

w1 , v1 = np.linalg.eig(transAA)
# w2, v2 = np.linalg.eig(AtransA)
f_vec = v1[:,8]

f_mat = np.reshape(f_vec, (3,3))

print(f_mat)
# %%
h_mat = np.zeros((2,2))
h_mat[0][0] = -f_mat[1][0]
h_mat[0][1] = -f_mat[1][1]
h_mat[1][0] = f_mat[0][0]
h_mat[1][1] = f_mat[0][1]

H = h_mat
print(H)
# %%

f_mat_T_f_mat = np.matmul(f_mat.transpose(), f_mat)
w3, v3 = np.linalg.eig(f_mat_T_f_mat)
f_vec = v3[:,2]
e1vec = f_vec/f_vec[2]

f_T_f_mat = np.matmul(f_mat, f_mat.transpose())
_ , v4 = np.linalg.eig(f_T_f_mat)
f_vec = v4[:,2]
e2vec = f_vec/f_vec[2]

print(e1vec, e2vec)

# %%
eig_F = np.linalg.eigvals(f_mat)
# print(eig_F)
line1 = np.matmul(f_mat, [502, 548, 1])
print([1262, 112, 1]@f_mat@[1131, 7, 1])
