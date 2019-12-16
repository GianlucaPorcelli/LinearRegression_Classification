import pandas as pd

from utili import *
from norm import *

#------------------------------- IMPORT CSV + MODIFICHE DATASET -------------------------------#

#Importiamo il csv e lo trasformiamo in una matrice
data = pd.read_csv("wine.csv", sep = ",", header=None)
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

#Visualizzazione del costo allo stato iniziale
print("Cost at iteration 0:", Cost(X, y, theta))

#Effettuo il Gradient Descent
#-------------------------------------------- BATCH -------------------------------------------#

theta, history = gradientDescent(X, y, theta, alpha, num_iters)

#----------------------------------------------------------------------------------------------#


#----------------------------------------- STOCHASTIC -----------------------------------------#

#theta, history = stochastic_grad_des(X, y, theta, alpha, num_iters)

#----------------------------------------------------------------------------------------------#


#----------------------------------------- MINI BATCH -----------------------------------------#

#Numero di minibatch da controllare
# b = 799
# theta, history = mini_batch(X, y, theta, alpha, num_iters, b)

#----------------------------------------------------------------------------------------------#

plotLearning(history)

#Inserisco la nuova tupla da predire
new_tuple = [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0] #inserire la tupla da predire


#Normalizzo la tupla da predire
#-------------------------------------- predizione zScore -------------------------------------#

new_tuple = zScore(new_tuple, mu, sigma)

#----------------------------------------------------------------------------------------------#


#-------------------------------------- predizione minmax -------------------------------------#

# new_tuple = np.asarray(new_tuple)
# new_tuple = Min_Max(new_tuple, min, diff)

#----------------------------------------------------------------------------------------------#

#----------------------------------- predizione feat scaling ----------------------------------#

# new_tuple = np.asarray(new_tuple)
# new_tuple = Feat_Scaling(new_tuple, max)

#----------------------------------------------------------------------------------------------#

#Aggiungo x0 alla tupla
new_tuple = np.insert(new_tuple, 0, 1)

#Effettuo la predizione
print("Prediction of tuple [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0] is:\n",predict(new_tuple, theta))