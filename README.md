# remote_sensing_satellite_image

Este repositório tem como principal objetivo mostrar uma série de técnicas de visão computacional aplicadas ao sensoreamento remoto em imagens satelitais para realizar tarefas como remoção de ruido em imagens e segmentação de pixels em diferentes cenários. Para isto foram utilizadas uma série bibliotecas tais como rasterio, gdal e earthpy a fim de realizar o tratamento e visualização dos dados satelitais.  

Para o processamento das imagens foram utilizadas técnicas clássicas de visão computacional como transformação de espaço de cores, pre e pós processamento de imagens e transformações morfológicas, além da utilização da arquitetura U-net baseada em deep learning para treinar um modelo que vise realizar a segmentação de residências em um conjunto de imagens satelitais.

```
- `database`: Contém arquivos associados à imagens das pessoas que estão cadastradas no sistema (considerando banco de dados);

- `util.py`: Script que contém as funções auxiliares para cadastro dos usuários;

- `main.py`: Script principal para executar o sistema de reconocimento facial. 
```
