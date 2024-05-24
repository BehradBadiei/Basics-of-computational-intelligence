import numpy as np
import matplotlib.pyplot as plt

def fill(i,j,img):
    a= 32*i
    b= 32*(i+1)
    c= 32*j
    d= 32 * (j+1)
    for x in range(a,b):
        for y in range(c,d):
            img[x,y] = 255
img = np.zeros([256,256])
for i in range(8):
    for j in range(8):
        if (i+j)%2==0:
            fill(i,j,img)            
plt.imshow(img,cmap='gray')
plt.show()          