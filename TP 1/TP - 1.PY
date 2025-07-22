# coding: utf-8

# # TP1: Développement d’un programme de régression linéaire.

# ## Etape 1 : Importer les librairies

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # We'll need this for Part 2

from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split # For Part 2, but good to have
from sklearn.metrics import r2_score # For evaluating R2 score
from sklearn.preprocessing import StandardScaler # Added for Part 2 feature scaling

# ## Etape 2 : Créer un Dataset

np.random.seed(0) # Pour la reproductibilité

x_generated, y_generated = make_regression(n_samples=100, n_features=1, noise=10)

plt.figure(figsize=(8, 6))
plt.scatter(x_generated, y_generated)
plt.title('Dataset Généré Aléatoirement')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# ## Etape 3 : Développer le modèle et l’entraîner

# #### Premier entraînement (SGDRegressor - mauvais hyperparamètres from notebook)
print("--- ETAPE 3: Développer le modèle et l'entraîner (SGDRegressor) ---")
print("--- Premier entraînement (mauvais hyperparamètres) ---")

# From cell 2326960a4faa144b
model_sgd_bad_nb = SGDRegressor(max_iter=100, eta0=0.0001) 
# The notebook cell produced a ConvergenceWarning. max_iter is low.
# Adding random_state for consistency if the algorithm has internal randomness not controlled by np.random.seed
# model_sgd_bad_nb = SGDRegressor(max_iter=100, eta0=0.0001, random_state=0) 
model_sgd_bad_nb.fit(x_generated, y_generated)

# From cell c04612e90665ffc9
y_pred_bad_nb = model_sgd_bad_nb.predict(x_generated)
r2_bad_nb = model_sgd_bad_nb.score(x_generated, y_generated) 

print(f'Coeff R2 (mauvais modèle - notebook) = {r2_bad_nb}')

# From cell 3e09e41a35230609
plt.figure(figsize=(8, 6))
plt.scatter(x_generated, y_generated, label='Données réelles')
plt.plot(x_generated, y_pred_bad_nb, c='red', lw=3, label='Prédictions (mauvais modèle - notebook)')
plt.title('Régression Linéaire avec Mauvais Hyperparamètres (Notebook)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# #### Ce résultat montre que notre modèle semble vraiment mauvais. C’est parce que nous nel’avons pas entraîné suffisamment longtemps et parce que le Learning rate était trop faible.

# ### Deuxième entraînement (bons hyperparamètres - from notebook)
print("\n--- Deuxième entraînement (bons hyperparamètres - from notebook) ---")

# From cell 6a3631e99e7522b3
# Note: The TP document suggests max_iter=1000, eta0=0.001.
# The notebook uses max_iter=10000, eta0=0.0001 and gets a good R2.
model_sgd_good_nb = SGDRegressor(max_iter=10000, eta0=0.0001, random_state=0) # Added random_state for consistency
model_sgd_good_nb.fit(x_generated, y_generated)

# From cell 9629e7a6eb2b45ad
y_pred_good_nb = model_sgd_good_nb.predict(x_generated)
r2_good_nb = model_sgd_good_nb.score(x_generated, y_generated) 

print(f'Coeff R2 (bon modèle - notebook) = {r2_good_nb}')

# From cell a6d00826135123d
plt.figure(figsize=(8, 6))
plt.scatter(x_generated, y_generated, label='Données réelles')
plt.plot(x_generated, y_pred_good_nb, c='red', lw=3, label='Prédictions (bon modèle - notebook)')
plt.title('Régression Linéaire avec "Bons" Hyperparamètres (Notebook)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# ### Vous avez entraîné votre premier modèle de Machine Learning, et il fonctionne vraiment bien avec un coefficient 𝑅2 = 94%. Vous pourriez maintenant vous en servir pour faire de bonnes prédictions.

# ## VISUALISATION DES COURBES D'APPRENTISSAGE (MANUELLE)

# ### Importer les librairies (already imported, but shown in notebook)
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_regression

# From cell e349c9000f10be1d (Générer un Dataset aléatoire - manual)
print("\n--- VISUALISATION DES COURBES D'APPRENTISSAGE (MANUELLE - Notebook) ---")
np.random.seed(4) # Nouveau seed comme dans le TP
m_manual_nb = 100 # Nombre d'échantillons
n_features_manual_nb = 1 # Nombre de features (x)

# From cell 5a94143017a9d9e5
x_manual_nb, y_manual_orig_nb = make_regression(n_samples=m_manual_nb, n_features=n_features_manual_nb, noise=10, random_state=4)
y_manual_nb = y_manual_orig_nb + 100

# From cell e50695f340cf6215
plt.figure(figsize=(8, 6))
plt.scatter(x_manual_nb, y_manual_nb)
plt.title('Dataset pour Descente de Gradient Manuelle (Notebook)')
plt.xlabel('x_manual_nb')
plt.ylabel('y_manual_nb')
plt.show()

# From cell a0793e4edc75e02e
# Préparer y_manual_nb pour les calculs matriciels (vecteur colonne)
y_manual_nb = y_manual_nb.reshape(y_manual_nb.shape[0], 1)

# From cell 9719cf8877ac8765
# Ajouter le Biais (colonne de 1s) à X
X_manual_b_nb = np.hstack((np.ones((m_manual_nb, 1)), x_manual_nb))
print(f"Shape of X_manual_b_nb (avec biais): {X_manual_b_nb.shape}") 
print(f"Shape of y_manual_nb: {y_manual_nb.shape}") 

# #### Définir le modèle, la Fonction Coût et le Gradient

# From cell 214cc006f2d96ad1
def model_func_nb(X, theta):
    return X.dot(theta)

def cost_function_nb(X, y, theta):
    m_samples = len(y)
    J = 1/(2*m_samples) * np.sum((model_func_nb(X, theta) - y)**2)
    return J

def gradient_nb(X, y, theta):
    m_samples = len(y)
    return 1/m_samples * X.T.dot(model_func_nb(X, theta) - y)

# #### Algorithme de Descente de Gradient

# From cell e43943e1bcacec21
def gradient_descent_nb(X, y, theta_init, learning_rate=0.001, iterations=1000): # default LR changed from 0.0001
    m_samples = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, theta_init.shape[0])) 

    theta = theta_init.copy() 

    for i in range(iterations):
        theta = theta - learning_rate * gradient_nb(X, y, theta)
        cost_history[i] = cost_function_nb(X, y, theta)
        theta_history[i,:] = theta.T 
    
    return theta, cost_history, theta_history

# #### Utilisation de l'algorithme et visualisation

# From cell d3aa9b31e63f541b
np.random.seed(0) 
theta_initial_nb = np.random.randn(X_manual_b_nb.shape[1], 1) 

iterations_manual_nb = 10 # As per notebook
learning_rate_manual_nb = 0.3 # As per notebook

theta_final_nb, cost_history_manual_nb, theta_history_manual_nb = gradient_descent_nb(
    X_manual_b_nb, y_manual_nb, theta_initial_nb, 
    learning_rate=learning_rate_manual_nb, 
    iterations=iterations_manual_nb
)

# From cell 200ebd03419aa962
print(f"\nTheta final (manuel - notebook): \n{theta_final_nb}")

plt.figure(figsize=(10, 6))
plt.plot(range(iterations_manual_nb), cost_history_manual_nb)
plt.title("Courbe d'Apprentissage (Fonction Coût J(Theta) - Notebook)")
plt.xlabel('Iterations')
plt.ylabel('J(Theta)')
plt.show()

# Visualisation du modèle au cours de son apprentissage (copied from previous response, adapted for notebook)
# The notebook itself does not have the code for this second plot in the "manual" section.
# This plot is shown on page 5 of the TP document.
# The code for this plot IS on page 7 of the TP document, under "# visualisation du modele au cours de son apprentissage"
# I will include it here as it's part of the TP.

plt.figure(figsize=(10, 6))
plt.plot(x_manual_nb, y_manual_nb, 'b.', label='Données réelles') 

plot_every_n_iterations_nb = max(1, iterations_manual_nb // 10) 
if iterations_manual_nb <= 10 : plot_every_n_iterations_nb = 1 # Ensure small iterations are plotted

for i in range(0, iterations_manual_nb, plot_every_n_iterations_nb):
    y_predict_iter_nb = model_func_nb(X_manual_b_nb, theta_history_manual_nb[i,:].reshape(-1,1))
    # For clarity in the plot legend, only label a few lines
    label_iter = f'Iter {i}' if i % (plot_every_n_iterations_nb * (10//iterations_manual_nb +1)) == 0 or i == 0 else None
    plt.plot(x_manual_nb, y_predict_iter_nb, lw=1, alpha=0.5, label=label_iter)

y_predict_final_nb = model_func_nb(X_manual_b_nb, theta_final_nb)
plt.plot(x_manual_nb, y_predict_final_nb, 'r-', lw=2, label=f'Modèle final (Iter {iterations_manual_nb})')

plt.title('Évolution du Modèle pendant la Descente de Gradient (Notebook)')
plt.xlabel('x_manual_nb')
plt.ylabel('y_manual_nb')
if iterations_manual_nb <= 10: # If few iterations, legend might be too cluttered
    plt.legend(loc='lower right', fontsize='small')
else:
    plt.legend()
plt.show()


# ## Partie 2 : Utilisation des données réelles importées depuis un fichier CSV
print("\n--- PARTIE 2: UTILISATION DES DONNÉES RÉELLES (FICHIER CSV) ---")

# ### INPORT CSV (Create the data file first for self-containment)
csv_data_content = """YearsExperience,Salary
1.1,39343.0
1.3,46205.0
1.5,37731.0
2.0,43525.0
2.2,39891.0
2.9,56642.0
3.0,60150.0
3.2,54445.0
3.2,64445.0
3.7,57189.0
3.9,63218.0
4.0,55794.0
4.0,57081.0
4.1,57081.0 # Assuming this was meant to be 57081, TP image is cut
4.5,61111.0
4.9,67938.0
5.1,66029.0
5.3,83088.0
5.9,81363.0
6.0,93940.0
6.8,91738.0
7.1,98273.0
7.9,101302.0
8.2,113812.0
8.7,109431.0
9.0,105582.0
9.5,116969.0
9.6,112635.0
10.3,122391.0
10.5,121872.0
"""
# The TP document on page 8 shows only up to "4 55794". 
# The notebook reads 'Data_salaire.csv' which likely contains more data.
# I'll use the full "Salary_Data.csv" that's standard for this example for better results.
# Correcting the filename from notebook to 'Salary_Data.csv'
csv_filename = 'Salary_Data.csv' 
with open(csv_filename, 'w') as f:
    f.write(csv_data_content)
print(f"\n'{csv_filename}' created/updated for Part 2 based on common Salary_Data.csv.")

# From cell 5cd78a3de66da0fe
dataset_real = pd.read_csv(csv_filename) # Using the newly created file
print("\nDataset chargé:")
print(dataset_real.head())

# ### Séparation des entrées « features » des sorties « Target »
# X_real: toutes les colonnes sauf la dernière
# y_real: la dernière colonne

# From cell 2e5cefd3993d7b0b
X_real_nb = dataset_real.iloc[:, :-1].values  # Features (YearsExperience)
y_real_nb = dataset_real.iloc[:, -1].values   # Target (Salary)

print(f"\nShape of X_real_nb (features): {X_real_nb.shape}")
print(f"Shape of y_real_nb (target): {y_real_nb.shape}")

# ### Répartition des données en entraînement et test
# test_size=0.2 signifie 20% pour le test, 80% pour l'entraînement
# random_state est pour la reproductibilité

# From cell 7ac0c0420210f8e1
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_real_nb, y_real_nb, test_size=0.2, random_state=42)

print(f"\nShape of X_train_nb: {X_train_nb.shape}, y_train_nb: {y_train_nb.shape}")
print(f"Shape of X_test_nb: {X_test_nb.shape}, y_test_nb: {y_test_nb.shape}")

# From cell 4fdfaf4048f2a5a8
# Entraîner le modèle (SGDRegressor) sur les données d'entraînement
# Using parameters from the notebook cell
model_real_data_nb = SGDRegressor(max_iter=10000, eta0=0.001, random_state=42) 

# From cell 3980fdb031131617 (Feature Scaling)
scaler_X_nb = StandardScaler()
X_train_scaled_nb = scaler_X_nb.fit_transform(X_train_nb)
X_test_scaled_nb = scaler_X_nb.transform(X_test_nb)

# From cell a5cce043050c3d14
print("\nEntraînement du modèle SGDRegressor sur les données réelles (X_train_scaled_nb)...")
model_real_data_nb.fit(X_train_scaled_nb, y_train_nb)

# From cell 9a5b1b6ab06f49c0
y_pred_real_test_scaled_nb = model_real_data_nb.predict(X_test_scaled_nb)

# From cell 6cb302f8e53445db
r2_real_test_nb = model_real_data_nb.score(X_test_scaled_nb, y_test_nb)

# From cell 7724f941efab9122
print(f"\nCoeff R2 sur les données de test (X_test_scaled_nb) = {r2_real_test_nb}")
print(f"Paramètres du modèle réel: Intercept = {model_real_data_nb.intercept_}, Coeffs = {model_real_data_nb.coef_}")

# ### Visualisation des résultats sur l'ensemble de test

# From cell 432fa67626af890a
plt.figure(figsize=(10, 6))
plt.scatter(X_test_scaled_nb, y_test_nb, color='blue', label='Données de Test Réelles (scaled X)')
plt.plot(X_test_scaled_nb, y_pred_real_test_scaled_nb, color='red', linewidth=2, label='Prédictions du Modèle')
plt.title('Régression Linéaire sur Données Réelles (Test Set - Scaled X - Notebook)')
plt.xlabel('YearsExperience (Scaled)')
plt.ylabel('Salary')
plt.legend()
plt.show()

# From the original TP response, plotting on original scale (not explicitly in notebook code but good practice)
plt.figure(figsize=(10, 6))
plt.scatter(X_test_nb, y_test_nb, color='blue', label='Données de Test Réelles')

# To plot the line on the original scale, we need to sort X_test for a continuous line
# or predict on a range of original X_train transformed
sorted_indices_nb = np.argsort(X_test_nb[:, 0])
X_test_sorted_nb = X_test_nb[sorted_indices_nb]
y_pred_for_plot_sorted_nb = model_real_data_nb.predict(scaler_X_nb.transform(X_test_sorted_nb))

plt.plot(X_test_sorted_nb, y_pred_for_plot_sorted_nb, color='red', linewidth=2, label='Prédictions du Modèle')
plt.title('Régression Linéaire sur Données Réelles (Test Set - Original X Scale - Notebook)')
plt.xlabel('YearsExperience (Original)')
plt.ylabel('Salary')
plt.legend()
plt.show()

print("\n--- FIN DU TP (Conversion Notebook) ---")