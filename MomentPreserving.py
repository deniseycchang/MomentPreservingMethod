import matplotlib.pyplot as plt
import cv2
#from PIL import Image
import numpy as np
import math
import cmath
from skimage import io
from skimage import color

def calculateA(c0, c1, c2):
    
    A1 = c0/2 - c1*c2/6 + (c2**3)/27
    A2 = cmath.sqrt(A1**2 + (c1/3 - (c2**2)/9)**3 )
    
    A = (A1 - A2)**(1./3.)

    return A

def calculateB(A, c1, c2):
    
    B = -(c1/3 - c2**2 / 9) / A
    
    return B

def momentpreserving(imgpath):

    N = 256
    
    imggray = color.rgb2gray(io.imread(imgpath))
    
    print(imggray.shape)
    
    (W, L) = imggray.shape
    
    
    img = cv2.imread(imgpath,0)
    
    img1D = img.ravel()
    plt.hist(img1D, N,[0, N]); plt.show()
    
    histimg, _ = np.histogram(img1D, bins=np.arange(257))
    
    m0 = 1
    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    m5 = 0
    
    for i in range(0, N):
        
        m1 = m1 + i * histimg[i]/len(img1D)
        m2 = m2 + i * i * histimg[i]/len(img1D)
        m3 = m3 + i * i * i * histimg[i]/len(img1D)
        m4 = m4 + i * i * i * i * histimg[i]/len(img1D)
        m5 = m5 + i* i * i * i * i * histimg[i]/len(img1D)
      
    ################## Bilevel ####################################
    
    cd = m0 * m2 - m1**2
    c0 = (-m2 * m2 + m1 * m3) / cd
    c1 = (m0 * -m3 + m2 * m1) / cd
    z0 = 0.5 * (-c1 - math.sqrt(c1 * c1 - 4.0 * c0))
    z1 = 0.5 * (-c1 + math.sqrt(c1 * c1 - 4.0 * c0))
    
    p0 = (z1 - m1) / (z1 - z0);
    
    partialsum = 0;
    
    print(p0)
    
    for i in range(0, N):
        partialsum += histimg[i]/len(img1D)
        if partialsum > p0:
            threshold = i
            break
    print(threshold)  
        
    pixgrey = np.zeros((W, L), 'uint8')
    
    
    for i in range(0, L):
        for j in range(0, W):
            if img[j, i] > threshold:
    
                pixgrey[j, i] = z1
            else:
                pixgrey[j, i] = z0
                    
    imgtitle = 'threshold: ' + str(threshold) + ', z0 = ' \
            + str(int(round(z0))) + ', z1 = ' + str(int(round(z1)))
    
    f, (ax0, ax1) = plt.subplots(2, 1, \
                                      subplot_kw={'xticks': [], 'yticks': []})
    
    ax0.imshow(imggray, cmap=plt.cm.gray)
    ax1.imshow(pixgrey, cmap=plt.cm.gray)
    ax0.set_title(imgtitle)
    f.set_size_inches(4, 6)
    
    ################## Trilevel ####################################
    
    cd = np.linalg.det(np.array([[m0, m1, m2],[m1, m2, m3],[m2, m3, m4]]))
    
    c0 = 1/cd * np.linalg.det(np.array([[-m3, m1, m2],\
                                        [-m4, m2, m3],\
                                        [-m5, m3, m4]]))
        
    c1 = 1/cd * np.linalg.det(np.array([[m0, -m3, m2],\
                                        [m1, -m4, m3],\
                                        [m2, -m5, m4]]))
        
    c2 = 1/cd * np.linalg.det(np.array([[m0, m1, -m3],\
                                        [m1, m2, -m4],\
                                        [m2, m3, -m5]]))
        
    A = calculateA(c0, c1, c2)
    
    #print('A: ', A)
    B = calculateB(A, c1, c2)
    
    #print('B: ', B)
    
    W1 = complex(-0.5, math.sqrt(3)/2)
        
    W2 = complex(-0.5, -math.sqrt(3)/2)
    
    
    z0 = -c2/3 - (A+B).real
    
    
    z1 = -c2/3 - (W1*A + W2*B).real
    
    
    z2 = -c2/3 - (W2*A + W1*B).real
    
    
    pd  = np.linalg.det([[1, 1, 1],[z0, z1, z2],[z0**2, z1**2, z2**2]])
    
    p0 = 1/pd * np.linalg.det([[m0, 1, 1],[m1, z1, z2],[m2, z1**2, z2**2]])
    
    p1 = 1/pd * np.linalg.det([[1, m0, 1],[z0, m1, z2],[z0**2, m2, z2**2]])
    
    partialsum = 0  
    
    for i in range(0, N):
        partialsum += histimg[i]/len(img1D)
        if partialsum > p0:
            threshold1 = i
            break
    
    partialsum = 0
    for i in range(threshold1, N):
        partialsum += histimg[i]/len(img1D)
        if partialsum > p1:
            threshold2 = i
            break
    
    pixtri = np.zeros(( W, L), 'uint8')
    
    for i in range(0, L):
        for j in range(0, W):
            if img[j, i] < threshold1:
                pixtri[j, i] = z0
            elif img[j, i] < threshold2:
    
                pixtri[j, i] = z1
            else:
                pixtri[j, i] = z2
    
    
    imgtitle = 'threshold1: ' + str(threshold1) + ', threshold2: ' + str(threshold2) \
            + ', \nz0 = ' + str(round(z0)) + ', z1 = ' + str(round(z1)) \
            + ', z2 = ' + str(round(z2))
    
    
    f2, (bx0, bx1) = plt.subplots(2, 1, \
                                      subplot_kw={'xticks': [], 'yticks': []})
    bx0.imshow(img, cmap=plt.cm.gray)
    bx1.imshow(pixtri, cmap=plt.cm.gray)
    bx0.set_title(imgtitle)
    f2.set_size_inches(4, 6)


#### -------------------------------------------------------------------------

imgpath = 'mountain-sky.jpg'
momentpreserving(imgpath)
