# %%
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

matlab_white = '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/rat/rat_points3d_cm.mat'
matlab_black = '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/ratBlack/rat_points3d_cm.mat'

# %% load matlab data
mat_white = scipy.io.loadmat(matlab_white)
mat_black = scipy.io.loadmat(matlab_black)
points_white = mat_white['points_3d']
points_black = mat_black['points_3d']

assert points_white.shape == points_black.shape, 'points_white and points_black should have the same shape'
assert points_white.shape[1:] == (14, 3) , 'points should be 14parts x 3xyz'   # 14 points, 3d
assert points_white.shape[0] > 5, 'points should be more than 5'

print('Accepted samples:', points_white.shape[0])

# %% 
z = points_white[:,[6,8],-1]
plt.subplot(1,2,1)
plt.plot(z) 

z = points_white[:,[10,12],-1]
plt.subplot(1,2,2)
plt.plot(z) 

# %%
""""
* findLinePlaneIntersectionCoords (to avoid requiring unnecessary instantiation)
* Given points p with px py pz and q that define a line, and the plane
* of formula ax+by+cz+d = 0, returns the intersection point or null if none.
"""
def findLinePlaneIntersectionCoords(P, Q, a=0, b=0, c=1, d=-1):
    px, py, pz = P[:,0], P[:,1], P[:,2]
    qx, qy, qz = Q[:,0], Q[:,1], Q[:,2]
    tDenom = a*(qx-px) + b*(qy-py) + c*(qz-pz)
    t = - ( a*px + b*py + c*pz + d ) / tDenom
    x, y, z = (px+t*(qx-px), py+t*(qy-py), pz+t*(qz-pz))
    xyz = np.array([x,y,z])
    qxyz = np.array([qx,qy,qz])
    mask_revise = a*qx + b*qy + c*qz + d < 0   #revise the point if under the plane
    rxyz = xyz * mask_revise + qxyz * (1-mask_revise)
    return rxyz.T

pshoulder = points_white[:,6,:]
ppaw = points_white[:,7,:]
ppaw2 = findLinePlaneIntersectionCoords(pshoulder, ppaw)

# %% paws
shoulderIndexs = [6,8,10,12]
pawIndexs = [7,9,11,13]

for shoulderIndex, pawIndex in zip(shoulderIndexs, pawIndexs):
    points3D = points_white
    shoulder, paw = points3D[:,shoulderIndex,:], points_white[:,pawIndex,:]
    paw2 = findLinePlaneIntersectionCoords(shoulder, paw)
    points3D[:,pawIndex,:] = paw2

    points3D = points_black
    shoulder, paw = points3D[:,shoulderIndex,:], points_black[:,pawIndex,:]
    paw2 = findLinePlaneIntersectionCoords(shoulder, paw)
    points3D[:,pawIndex,:] = paw2


# %% head or tail, constrain by two lines
pointIndexs = [0,5]
anchersIndexs = [[1,2],[10,12]]

for pointIndex, anchersIndex in zip(pointIndexs, anchersIndexs):
    points3D = points_white
    point, ancher1, ancher2 = points3D[:,pointIndex,:], points3D[:,anchersIndex[0],:], points3D[:,anchersIndex[1],:]
    point_anchers1 = findLinePlaneIntersectionCoords(point, ancher1, d=0)
    point_anchers2 = findLinePlaneIntersectionCoords(point, ancher2, d=0)
    points3D[:,pointIndex,:] = (point_anchers1 + point_anchers2)/2

    points3D = points_black
    point, ancher1, ancher2 = points3D[:,pointIndex,:], points3D[:,anchersIndex[0],:], points3D[:,anchersIndex[1],:]
    point_anchers1 = findLinePlaneIntersectionCoords(point, ancher1, d=0)
    point_anchers2 = findLinePlaneIntersectionCoords(point, ancher2, d=0)
    points3D[:,pointIndex,:] = (point_anchers1 + point_anchers2)/2





# %%
plt.figure()
i_sample = 56
p1 = pshoulder[i_sample,[1,2]]  # x,z
p2 = ppaw[i_sample,[1,2]]
p3 = ppaw2[i_sample,[1,2]]
p1p2 = np.array([p1, p2])
p2p3 = np.array([p2, p3])
plt.plot(p1p2[:,0], p1p2[:,1], '-o')
plt.plot(p2p3[:,0], p2p3[:,1], '-rx')
# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
# 指定渲染环境
%matplotlib notebook
# %matplotlib inline
 
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
 
fig = plt.figure(tight_layout=True)
plt.plot(x,y)
plt.grid(ls="--")
plt.show()
# %%
