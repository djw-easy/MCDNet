import os
import cv2
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


root_dir = './data/l8'
for split in ['Train', 'Test']:
    in_dir = os.path.join(root_dir, split, 'cloudy')
    out_dir = os.path.join(root_dir, split, 'bccr')
    os.makedirs(out_dir, exist_ok=True)
    for file_name in os.listdir(in_dir):
        if not file_name.endswith('.TIF'):
            continue
        out_path = os.path.join(out_dir, file_name)
        # if os.path.exists(out_path):
        #     continue
        in_path = os.path.join(in_dir, file_name)
        src = rio.open(in_path)
        HazeImg = cv2.imread(in_path)
        HazeImg = np.flip(HazeImg, axis=2)
        HazeCorrectedImg, _ = image_dehazer.remove_haze(HazeImg, showHazeTransmissionMap=False)
        array_to_tif(out_path, np.transpose(HazeCorrectedImg, axes=(2, 0, 1)), src.crs, src.transform)

