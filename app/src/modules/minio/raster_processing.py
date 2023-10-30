import rasterio
import numpy as np
from rasterio.plot import show
from osgeo import gdal
from matplotlib import gridspec
import cv2


def raster_processing_img(raster_path): 
    '''Função para processamento de raster
    
    Args: 
    
    raster_path = caminho relativo que contem a imagem satelital com 8 bandas. 
    
    Return:
    
    Bandas 4, 3, 2 para cálculo de imagem RGB em outros programas
    
    Imagem RGB combinando as bandas 4,3,2
    
    Banda 5 para cálculo de NDVI.
    
    ''' 
    raster = rasterio.open(path)
    
    print("Count: ",raster.count)
    print("Height and widht: ", raster.height ,raster.width)
    print("CRS", raster.crs)
    print("bounds: ",raster.bounds)
    
    band4=raster.read(4)
    band3=raster.read(3)
    band2=raster.read(2)

    band5 = raster.read(5)
   
    band4=np.array(band4)
    band3=np.array(band3)
    band2=np.array(band2)
    
    rgb = np.dstack((band4,band3,band2))

    # Min-Max scaling
    min_val = np.min(rgb)
    max_val = np.max(rgb)
    scaled_data = (rgb - min_val) / (max_val - min_val)
    plt.imshow(scaled_data)
    #plt.imshow(rgb)
    scaled_data_rgb = 255 * scaled_data
    rgb = scaled_data_rgb.astype('uint8')
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    
    return rgb, band2, band3, band4, band5 
    