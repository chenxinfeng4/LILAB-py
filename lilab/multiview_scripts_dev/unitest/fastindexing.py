#%%
import numpy as np
import tqdm

#%%
a = np.random.randint(0,255,[800*3,1280*3,3], dtype=np.uint8)
b = np.random.randint(0,800*3*1280*3*1,[9, 1,320, 512], dtype=np.int64)
d = np.arange(9*1*320*512).reshape([9, 1,320, 512])
#%%
for _ in tqdm.trange(1000):
    c = a.ravel()[b]
    c_cp = np.repeat(c, 3, 1)
# %%
d[:]=0
# d = np.arange(9*1*320*512).reshape([9, 1,320, 512])
# d = np.random.randint(0,800*3,[9, 1,320, 512], dtype=np.int64)

d = np.random.randint(0,800*3*1280*3*3,[9, 1,320, 512], dtype=np.int64)
for _ in tqdm.trange(1000):
    c = a.ravel()[d]

for _ in tqdm.trange(1000):
    x = np.ascontiguousarray(a[:800,:1280,:])
    x = np.ascontiguousarray(a[800:1600,1280:1280*2,:])
    x = np.ascontiguousarray(a[800*2:800*3,1280:1280*2,:])
    x = np.ascontiguousarray(a[:800,:1280,:])
    x = np.ascontiguousarray(a[800:1600,1280:1280*2,:])
    x = np.ascontiguousarray(a[800*2:800*3,1280:1280*2,:])
    x = np.ascontiguousarray(a[:800,:1280,:])
    x = np.ascontiguousarray(a[800:1600,1280:1280*2,:])
    x = np.ascontiguousarray(a[800*2:800*3,1280:1280*2,:])

data = np.random.randint(0,800*3*1280*3*3,(9,1,360,512))
# data.ravel().sort()
for _ in tqdm.trange(1000):
    ph=3;pw=3
    h=360;w=512; c=1;
    data2 = data.reshape(ph,pw,c,h,w) # ->(ph,pw,c,h,w)
    data3 = np.transpose(data2, (0,3,1,4,2)) # ->(ph,h,pw,w,c)
    data4 = np.reshape(data3, (ph*h, pw*w, c))
    data4.ravel().sort()

a.ravel().sort()
for _ in tqdm.trange(1000):
    datacl = a.ravel()[data4]
    undata3 = datacl.reshape(ph,h,pw,w,c) # ->(ph,h,pw,w,c)
    undata2 = np.transpose(undata3, (0,2,4,1,3)) # ->(ph,pw,c,h,w)
    undata1 = undata2.reshape(ph*pw,c,h,w)
    undatac = np.repeat(undata1, 3, 1)

import matplotlib.pyplot as plt
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.title(f'{i}')
    plt.imshow(undata1[i,0])
    plt.clim(undata1.min(), undata1.max())