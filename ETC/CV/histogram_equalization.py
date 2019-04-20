import numpy as np 
import matplotlib.pyplot as plt 
from skimage import exposure as ex 
import imageio 
import sys 
import cv2     

import matplotlib.colors
import scipy, scipy.misc, scipy.signal

''' 
    caip_2017 Histogram Equalization 
    low speed, but high quality
'''
def computeTextureWeights(fin, sigma, sharpness):
    dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0,:]-fin[-1,:]))
    dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:,0].conj().T-fin[:,-1].conj().T)).conj().T

    gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1,sigma)), mode='same')
    gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma,1)), mode='same')

    W_h = 1/(np.abs(gauker_h)*np.abs(dt0_h)+sharpness)
    W_v = 1/(np.abs(gauker_v)*np.abs(dt0_v)+sharpness)

    return  W_h, W_v
    
def solveLinearEquation(IN, wx, wy, lamda):
    [r, c] = IN.shape
    k = r * c
    dx =  -lamda * wx.flatten('F')
    dy =  -lamda * wy.flatten('F')
    tempx = np.roll(wx, 1, axis=1)
    tempy = np.roll(wy, 1, axis=0)
    dxa = -lamda *tempx.flatten('F')
    dya = -lamda *tempy.flatten('F')
    tmp = wx[:,-1]
    tempx = np.concatenate((tmp[:,None], np.zeros((r,c-1))), axis=1)
    tmp = wy[-1,:]
    tempy = np.concatenate((tmp[None,:], np.zeros((r-1,c))), axis=0)
    dxd1 = -lamda * tempx.flatten('F')
    dyd1 = -lamda * tempy.flatten('F')
    
    wx[:,-1] = 0
    wy[-1,:] = 0
    dxd2 = -lamda * wx.flatten('F')
    dyd2 = -lamda * wy.flatten('F')
    
    Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:,None], dxd2[:,None]), axis=1).T, np.array([-k+r,-r]), k, k)
    Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None,:], dyd2[None,:]), axis=0), np.array([-r+1,-1]), k, k)
    D = 1 - ( dx + dy + dxa + dya)
    A = ((Ax+Ay) + (Ax+Ay).conj().T + scipy.sparse.spdiags(D, 0, k, k)).T
    
    tin = IN[:,:]
    tout = scipy.sparse.linalg.spsolve(A, tin.flatten('F'))
    OUT = np.reshape(tout, (r, c), order='F')
    
    return OUT
    

def tsmooth(img, lamda=0.01, sigma=3.0, sharpness=0.001):
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    x = np.copy(I)
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lamda)
    return S

def rgb2gm(I):
    if (I.shape[2] == 3):
        I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        I = (I[:,:,0]*I[:,:,1]*I[:,:,2])**(1/3)

    return I

def applyK(I, k, a=-0.3293, b=1.1258):
    f = lambda x: np.exp((1-x**a)*b)
    beta = f(k)
    gamma = k**a
    J = (I**gamma)*beta
    return J

def entropy(X):
    tmp = X * 255
    tmp[tmp > 255] = 255
    tmp[tmp<0] = 0
    tmp = tmp.astype(np.uint8)
    _, counts = np.unique(tmp, return_counts=True)
    pk = np.asarray(counts)
    pk = 1.0*pk / np.sum(pk, axis=0)
    S = -np.sum(pk * np.log2(pk), axis=0)
    return S

def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):
    # Esatimate k
    tmp = cv2.resize(I, (50,50), interpolation=cv2.INTER_AREA)
    tmp[tmp<0] = 0
    tmp = tmp.real
    Y = rgb2gm(tmp)
    
    isBad = isBad * 1
    isBad = scipy.misc.imresize(isBad, (50,50), interp='bicubic', mode='F')
    isBad[isBad<0.5] = 0
    isBad[isBad>=0.5] = 1
    Y = Y[isBad==1]
    
    if Y.size == 0:
       J = I
       return J
    
    f = lambda k: -entropy(applyK(Y, k))
    opt_k = scipy.optimize.fminbound(f, 1, 7)
    
    # Apply k
    J = applyK(I, opt_k, a, b) - 0.01
    return J

''' '''



def draw_histogram_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def draw_histogram(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()

def he(img):
    ''' equal cv2 '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outImg = np.zeros((img.shape[0], img.shape[1], 3))
    for channel in range(img.shape[2]):
        outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255
    
    outImg[outImg>255] = 255
    outImg[outImg<0] = 0
    outImg = cv2.cvtColor(outImg.astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imshow('Histogram equalized', outImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return outImg

def he_rgb(img):
    ''' 
        img : 이미지
        channel : grayscale = [0], color(bgr) = [0] or [1] or [2]
        mask : 이미지 전체 = None, 특정 영역 = 해당하는 mask 값 입력
        histSize : BIN 개수, []로 둘러싸기
        range : 픽셀값 범위 
    '''
    channels = cv2.split(img)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv2.equalizeHist(ch))

    eq_image = cv2.merge(eq_channels)
    cv2.imshow('Histogram equalized', eq_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return eq_image

def he_hsv(img):
    ''' '''
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)

    eq_image_bgr = cv2.cvtColor(eq_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Histogram equalized', eq_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plt.imshow(eq_image) # RGB
    # plt.show()

    return eq_image_bgr

def clahe(img):
    ''' '''
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # res = np.hstack((img, bgr))

    cv2.imshow('clahe result', bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return bgr

def caip_2017(img, mu=0.5, a=-0.3293, b=1.1258):
    ''' 
        Input : cv.imread('img.jpg', cv.IMREAD_COLOR)
        Output : histogram equalization applied image
    ''' 
    lamda = 0.5
    sigma = 5
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Weight matrix estimation
    t_b = np.max(I, axis=2)
    t_our = cv2.resize(tsmooth(scipy.misc.imresize(t_b, 0.5, interp='bicubic', mode='F'), 
        lamda, sigma), (t_b.shape[1], t_b.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Apply camera model with k(exposure ratio)
    isBad = t_our < 0.5
    J = maxEntropyEnhance(I, isBad)

    # W: Weight Matrix
    t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
    for i in range(I.shape[2]):
        t[:,:,i] = t_our
    W = t**mu

    I2 = I*W
    J2 = J*(1-W)

    result = I2 + J2
    result = result * 255
    result[result > 255] = 255
    result[result<0] = 0
    result = result.astype(np.uint8)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imshow('caip_2017 result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result

if __name__ == '__main__':
    # img_name = sys.argv[1]
    img = cv2.imread('res/img_10.jpg', cv2.IMREAD_COLOR)
    
    #draw_histogram(img)
    #draw_histogram_gray(img)
    eq_img = caip_2017(img) # he_rgb(img) # he(img) # clahe(img) # he_rgb(img) # he_hsv(img)
    #draw_histogram(eq_img)
    #draw_histogram_gray(eq_img)
