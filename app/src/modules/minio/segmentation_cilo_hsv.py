import cv2
import matplotlib.pyplot as plt
import numpy as np

def segment_hsv_img(path_image_raw):
    
    '''
    Função que tem como objetivo segmentar a estrutura "cilo" em imagens RGB
    
    Args: 
    
    imagem rgb = caminho relativo que contem a imagem satelital em formato RGB. 
    
    Return:
    
    Mascara binaria onde o conjunto de pixels com valor 1 representa a classe "cilo".
    
    '''
    
    #path = '/home/alex/Desktop/prova_senai/Q2/img67.png'
    out_path = '/home/alex/Desktop/prova_senai/Q2/img67_hsv.png'

    img = cv2.imread(path_image_raw)
    img_rgb = img.copy()
    w, h, _ = img.shape
    new_image = np.zeros((w, h))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 12, 32])
    upper = np.array([28, 99, 80])

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
        if area > 200:
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)

            kernel = np.ones((3, 3), np.uint8)
            kernel_closing = np.ones((5, 5), np.uint8)
            output = cv2.dilate(output, kernel, iterations=1)
            output_mask = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel_closing)

    #cv2.imwrite(out_path, output_mask)
    
    return output_mask
    