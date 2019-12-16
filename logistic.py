import pandas as pd
#import numpy as np

#Per matplot
from utili import *
from norm import *

#------------------------------- IMPORT CSV + MODIFICHE DATASET -------------------------------#

#Importiamo il csv e lo trasformiamo in una matrice
data = pd.read_csv("candy_1.csv", sep = ",", header=None)
data = data.as_matrix()
#Creiamo le matrici x e y
X = data[:, 0:data.shape[1]-1]
y = data[:, -1]

#----------------------------------------------------------------------------------------------#






#------------------------------------ NORMALIZAZZIONE FEATURE ---------------------------------#

#------------------------------------- ZSCORE -----------------------------------#


mu, sigma = muSigma(X)
X = zScore(X, mu, sigma)

#--------------------------------------------------------------------------------#


#------------------------------------ MINMAX ------------------------------------#

# min, diff, max = minmax(X)
# X = Min_Max(X, min, diff)

#--------------------------------------------------------------------------------#


#-------------------------------- FEATURE SCALING -------------------------------#

# min, diff, max = minmax(X)
# X = Feat_Scaling(X, max)

#--------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------#



#Aggiungiamo x0 alle x
X = np.column_stack((np.ones(data.shape[0]), X))

#Genero theta
theta = np.zeros(X.shape[1])


#Parametri
alpha = 0.02
num_iters = 1000

z = predict(X, theta)
h = 1 / (1 + np.exp(-z))
print("Costo iniziale: ",CostLog(h, y))

#Effettuo il Gradient Descent
theta, history = gradientDescent_logistic(X, y, theta, alpha, num_iters)

z = predict(X, theta)
h = 1 / (1 + np.exp(-z))
print("Costo finale: ",CostLog(h, y))

#theta2 = normalEquations(X, y)


plotLearning(history)

#Inserisco la nuova tupla da predire
new_tuple = [0,1,0,0,0,0,1,0.31299999,0.51099998,23.417824] #inserire la tupla da predire

#Normalizzo la tupla da predire
new_tuple = zScore(new_tuple, mu, sigma)

#Aggiungo x0 alla tupla
new_tuple = np.insert(new_tuple, 0, 1)

#Effettuo la predizione
predictLog(theta, new_tuple, 1)