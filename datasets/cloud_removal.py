import os
import cv2, scipy
import numpy as np
import image_dehazer
import rasterio as rio


def array_to_tif(out_path, arr, crs, transform, nodata=None):
    # 获取数组的形状
    if arr.ndim==2:
        count = 1
        height, width = arr.shape
    elif arr.ndim==3:
        count = arr.shape[0]
        _, height, width = arr.shape
    else:
        raise ValueError
    
    with rio.open(out_path, 'w', 
                  driver='GTiff', 
                  height=height, width=width, 
                  count=count, 
                  dtype=arr.dtype, 
                  crs=crs, 
                  transform=transform, 
                  nodata=nodata) as dst:
        # 写入数据到输出文件
        if count==1:
            dst.write(arr, 1)
        else:
            for i in range(count):
                dst.write(arr[i, ...], i+1)


# Thin Cloud Removal using dark channel prior (DCP)
def zmMinFilterGray(src, size=7):
    '''''最小值滤波，r是滤波器半径'''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(src, kernel)
    return dark_channel

def guidedfilter(I, p, size, eps):
    '''''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (size, size))
    m_p = cv2.boxFilter(p, -1, (size, size))
    m_Ip = cv2.boxFilter(I*p, -1, (size, size))
    cov_Ip = m_Ip-m_I*m_p

    m_II = cv2.boxFilter(I*I, -1, (size, size))
    var_I = m_II-m_I*m_I

    a = cov_Ip/(var_I+eps)
    b = m_p-a*m_I

    m_a = cv2.boxFilter(a, -1, (size, size))
    m_b = cv2.boxFilter(b, -1, (size, size))
    return m_a*I+m_b

def getV1(m, size, guided_size, eps, w, maxV1):  #输入rgb图像，值范围[0,1]
    '''''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                                         #得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1, size), guided_size, eps)     #使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                              #计算大气光照A
    d = np.cumsum(ht[0])/float(V1.size)
    for lmax in range(bins-1, 0, -1):
        if d[lmax]<=0.999:
            break
    A  = np.mean(m, 2)[V1>=ht[1][lmax]].max()

    V1 = np.minimum(V1*w, maxV1)                   #对值范围进行限制

    return V1, A

def dcp_deHaze(m, size=7, guided_size=56, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    m = m / 255.0
    Y = np.zeros(m.shape)
    V1, A = getV1(m, size, guided_size, eps, w, maxV1)               #得到遮罩图像和大气光照
    for k in range(m.shape[-1]):
        Y[:, :, k] = (m[:, :, k]-V1)/(1-V1/A)           #颜色校正
    Y =  np.clip(Y, 0, 1)
    if bGamma:
        Y = Y**(np.log(0.5)/np.log(Y.mean()))       #gamma校正,默认不进行该操作
    
    return np.uint8(Y * 255.0)


# Thin Cloud Removal using momorphic filtering (HF)
def hf_deHaze(img):
    out = []
    rows = img.shape[0]
    cols = img.shape[1]
    for i in range(img.shape[-1]):
        imgLog = np.log1p(np.array(img[..., i], dtype="float") / 255)
        # 创建高斯掩膜
        M = 2*rows + 1
        N = 2*cols + 1
        sigma = 10
        (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
        centerX = np.ceil(N/2)
        centerY = np.ceil(M/2)
        gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

        # 低通和高通滤波器
        Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
        Hhigh = 1 - Hlow

        HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
        HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

        If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
        Ioutlow = np.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
        Iouthigh = np.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

        gamma1 = 0.3
        gamma2 = 1.5
        Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows, 0:cols]

        Ihmf = np.expm1(Iout)
        Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
        Ihmf2 = np.array(255.0*Ihmf, dtype="uint8")

        out.append(Ihmf2[..., np.newaxis])
    out = np.concatenate(out, axis=2)
    out = np.flip(out, axis=2)
    return out


def bccr_deHaze(img):
    out = image_dehazer.remove_haze(img, showHazeTransmissionMap=False)
    return out


