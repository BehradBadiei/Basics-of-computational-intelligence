import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = np.array(Image.open('malf.jpg'))
img = np.mean(img, axis=2)
plt.imshow(img, cmap='gray')
plt.title('Orginal Image')

def salt(img,selected):
    for i in range(len(selected)):
        ranDom = np.random.rand()
        if (ranDom <0.5):
            img[selected[i][0]][selected[i][1]] = 255
        else :
            img[selected[i][0]][selected[i][1]] = 0
    return img

def selectRandom(img,percent):
    percent = percent/100
    x,y = img.shape
    allindex = [(i,j)for i in range(x) for j in range(y)]
    count = int(len(allindex)*percent)
    randomindex = np.random.choice(len(allindex),count,replace=False)
    selected = [allindex[l] for l in randomindex]
    salted = salt(img,selected)
    return salted

percentage = 10
noised10 = selectRandom(img,percentage)
plt.imshow(noised10 , cmap='gray')
plt.title(f' {percentage}% Salt-And-Pepper Image')

percentage = 25
noised25 = selectRandom(img,percentage)
plt.imshow(noised25 , cmap='gray')
plt.title(f' {percentage}% Salt-And-Pepper Image')

percentage = 50
noised50 = selectRandom(img,percentage)
plt.imshow(noised50 , cmap='gray')
plt.title(f' {percentage}% Salt-And-Pepper Image')

def medianBlur(img):
    kernel = 9
    padSize = kernel // 2
    padding = np.pad(img, padSize, mode='constant', constant_values=0)
    outImg = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            selectMed = padding[i:i+kernel, j:j+kernel]
            outImg[i, j] = np.median(selectMed)
    
    return outImg

deNoise = medianBlur(noised50)
img2 = np.array(Image.open('malf.jpg'))
img2 = np.mean(img2, axis=2)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img2,cmap='gray')
axes[0].set_title('Orginal')
axes[0].axis('off')  

axes[1].imshow(noised50,cmap='gray')
axes[1].set_title('50% Noise')
axes[1].axis('off')  

axes[2].imshow(deNoise,cmap='gray')
axes[2].set_title('Median Blur')
axes[2].axis('off') 

plt.show()

