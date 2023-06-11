#%%
import os
import cv2 as cv
import numpy as np
import pandas as pd

pasta_imagens = 'frutasdb'

arquivos = os.listdir(pasta_imagens)

imagens = [arquivo for arquivo in arquivos if arquivo.endswith(('.bmp'))]

perimetros = []
diametros = []
nome_fruta = []
numero = []

tipo_doc = ".bmp"

for imagem in imagens:
    caminho_imagem = os.path.join(pasta_imagens, imagem)
    
    img1 = cv.cvtColor(cv.imread(caminho_imagem), cv.COLOR_BGR2GRAY)
    _, img1 = cv.threshold(img1, 120, 255, cv.THRESH_BINARY_INV)

    contorno,ordem = cv.findContours(img1,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    perimetro = cv.arcLength(contorno[0],True)
    perimetros.append(perimetro)

    diametro = np.sqrt(4*cv.contourArea(contorno[0])/np.pi)
    diametros.append(diametro)

    for i in range(0, len(tipo_doc)):
        imagem = imagem.replace(tipo_doc[i], "")

    numero.append(imagem)

    if int(imagem) <= 30:
        nome_fruta.append("Maçã")
    elif int(imagem) <= 60:
        nome_fruta.append("Abacaxi")
    elif int(imagem) <= 90:
        nome_fruta.append("Banana")
    elif int(imagem) <= 120:
        nome_fruta.append("Pêssego")
    elif int(imagem) <= 150:
        nome_fruta.append("Pitanga")
    elif int(imagem) <= 180:
        nome_fruta.append("Laranja")
    elif int(imagem) <= 210:
        nome_fruta.append("Morango")
    elif int(imagem) <= 240:
        nome_fruta.append("Pera")
    elif int(imagem) <= 270:
        nome_fruta.append("Limão")
    elif int(imagem) <= 300:
        nome_fruta.append("Uva")

dados = {'#': numero,
         'perimetro': perimetros,
         'diametro': diametros}
df = pd.DataFrame(dados)
nome_arquivo = 'dados.xlsx'
df.to_excel(nome_arquivo)
# %%
