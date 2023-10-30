import cv2
import matplotlib.pyplot as plt
import numpy as np
from raster_processing import raster_processing_img

def segment_vegetation_hsv_img(path_raster_raw):
    
    '''
    Função que tem como objetivo segmentar a classe "vegetação" em imagens satelitais
    
    Args: 
    
    path_image_raw = caminho relativo que contem a imagem satelital com um numero de bandas determinado (raster). 
    
    Return:
    
    Mascara binaria onde o conjunto de pixels com valor 1 representa a classe "vegetação".
    
    '''
    
    #path = '/home/alex/Desktop/prova_senai/Q2/img67.png'
    #out_path = '/home/alex/Desktop/prova_senai/Q2/img67_hsv.png'

    rgb, band2, band3, band4, band5 = raster_processing_img(path_raster_raw)
    
    img_rgb = rgb.copy()
    img = cv2.medianBlur(img, 3)
    w, h, _ = img.shape
    new_image = np.zeros((w, h))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([17, 44, 0])
    upper = np.array([133, 192, 55])

    imgMASK = cv2.inRange(imgHSV, lower, upper)
    bitwiseAnd = cv2.bitwise_not(imgMASK)

    mask_post  = imgMASK.copy()

    new_pre = np.zeros_like(imgMASK, dtype="uint8")
    new_pre[mask_post != 0] = 255

    analysis = cv2.connectedComponentsWithStats(new_pre, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    output = np.zeros(mask_post.shape, dtype="uint8")

    # Loop through each component
    for i in range(1, totalLabels):
        # Area of the component
        area = values[i, cv2.CC_STAT_AREA] 
        if area > 100:
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)

    #cv2.imwrite(out_path, output_mask)
    
    return output_mask
    