import pandas as pd
#import numpy as np

#Per matplot
from utili import *
from norm import *


#------------------------------- IMPORT CSV + MODIFICHE DATASET -------------------------------#

#Importiamo il csv e lo trasformiamo in una matrice
data = pd.read_csv("candy_2.csv", sep = ",", header=None)
data = data.as_matrix()
#Creiamo le matrici x e y
X = data[:, 0:data.shape[1]-3]
y = data[:, -3:]

#----------------------------------------------------------------------------------------------#







#------------------------------------ NORMALIZAZZIONE FEATURE ---------------------------------#

#------------------------------------- ZSCORE -----------------------------------#

#
# mu, sigma = muSigma(X)
# X = zScore(X, mu, sigma)

#--------------------------------------------------------------------------------#


#------------------------------------ MINMAX ------------------------------------#

min, diff, max = minmax(X)
X = Min_Max(X, min, diff)

#--------------------------------------------------------------------------------#


#-------------------------------- FEATURE SCALING -------------------------------#

# min, diff, max = minmax(X)
# X = Feat_Scaling(X, max)

#--------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------#



#Aggiungiamo x0 alle x
X = np.column_stack((np.ones(data.shape[0]), X))

#Genero theta
theta = np.zeros((X.shape[1], 3))


#Parametri
alpha = 0.02
num_iters = 100000



#Effettuo il Gradient Descent
theta, history = gradientDescent_logistic_multival(X, y, theta, alpha, num_iters)
#theta, history = stochastic_grad_des(X, y, theta, alpha, num_iters)



# theta = normalEquations(X, y)



for i in range(0,3):
    plotLearning(history[i][0])



#Inserisco la nuova tupla da predire
new_tuple = [0,0,0,1,0,1,0,0.31299999,0.51099998,76.7686] #inserire la tupla da predire

#Normalizzo la tupla da predire
#new_tuple = zScore(new_tuple, mu, sigma)

#Predico con min_max
new_tuple = np.asarray(new_tuple)
new_tuple = Min_Max(new_tuple, min, diff)


#Aggiungo x0 alla tupla
new_tuple = np.insert(new_tuple, 0, 1)

#Effettuo la predizione
prob = []

for i in range(0, y.shape[1]):
    prob.append([format(predictLog(theta[:,i], new_tuple, 2)*100), i])
    print("Appartiene alla classe ",prob[i][1]," con probabilita': ",prob[i][0],"%")
massimo = max(prob)
print("\n\nClasse predetta numero ",massimo[1]," con probabilita' di successo del ",massimo[0],"%")
