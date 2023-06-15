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

    #
    # DESCRITORES DE BORDA
    #
    contorno,ordem = cv.findContours(img1,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    perimetro = cv.arcLength(contorno[0],True)
    perimetros.append(perimetro)

    diametro = np.sqrt(4*cv.contourArea(contorno[0])/np.pi)
    diametros.append(diametro)

    #
    # DESCRITORES DE REGIÃO
    #
    area = cv.countNonZero(img1)

    compacidade = np.square(perimetro)/area

    # Momentos de Hu
    momentos = cv.moments(img1)
    momentos_hu = cv.HuMoments(momentos)

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
         'diametro': diametros,
         'frutas':nome_fruta}
df = pd.DataFrame(dados)
nome_arquivo = 'dados.xlsx'
df.to_excel(nome_arquivo)
# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#importar o dataset
dados = pd.read_excel('dados.xlsx')

#separas as 2 (duas) características e os nomes das espécies (classes)
X = dados.iloc[:,1:3]   # colunas de zero até três
y = dados.iloc[:,3]     # coluna quatro

#
# PRÉ-PROCESSAMENTO
#
normalizar = StandardScaler()
normalizar.fit(X)
X = normalizar.transform(X)

XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size=0.3)

print('Tamanho do dataset de TRAIN: {0}'.format(len(XTrain)))
print('Tamanho do dataset de TEST: {0}'.format(len(XTest)))

#
# CLASSIFICADOR
#
knn = KNeighborsClassifier(n_neighbors=3)
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(yTrain)
knn.fit(XTrain,y_transformed)
Y = knn.predict(XTest).round()

# RESULTADO
acc = accuracy_score(yTest,Y)
print('Acurácia: {0:.2f} ({1:.2f}%) \n\n'.format(acc,(acc*100)))

# MATRIZ CONFUSÃO
print(pd.crosstab(yTest,Y,rownames=['True'],colnames=['Predição'],margins=True))
# %%
