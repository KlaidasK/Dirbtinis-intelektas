import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

# sėklos reikšmė atkuriamumui
np.random.seed(12)

# Generuojama pirma klasė (žalia) (nuo 1 iki 5)
class1_x = np.random.uniform(1, 5, 10)
class1_y = np.random.uniform(1, 5, 10)
class1 = np.column_stack((class1_x, class1_y))

# Generuojama antra klasė (violetinė) (nuo 6 iki 10)
class2_x = np.random.uniform(6, 10, 10)
class2_y = np.random.uniform(6, 10, 10)
class2 = np.column_stack((class2_x, class2_y))

# Sujungiami duomenys
labels = np.array([0] * 10 + [1] * 10)  # 0 - pirma klasė, 1 - antra klasė
data = np.vstack((class1, class2))

# Vizualizuojame duomenis
plt.figure(figsize=(6, 6))
plt.scatter(class1[:, 0], class1[:, 1], color='green', label='Klasė 0')
plt.scatter(class2[:, 0], class2[:, 1], color='purple', label='Klasė 1')
plt.xlabel("X ašis")
plt.ylabel("Y ašis")
plt.title("Sugeneruoti duomenys")
plt.legend()
plt.grid(True)
plt.show()

# Išspausdinti taškų koordinates ir klases
for (x, y), label in zip(data, labels):
    print(f"Koordinatės: ({x:.2f}, {y:.2f}), Klasė: {label}")
    
    
# Neurono įėjimo duomenys
X = data

# Funkcija apskaičiuoti aktyvaciją
def neuron_activation(X, w, b, activation):
    a = np.dot(X, w) + b
    if activation == 'step':
        return np.where(a >= 0, 1, 0)
    elif activation == 'sigmoid':
        return np.round(1 / (1 + np.exp(-a)))
    else:
        raise ValueError("Nežinoma aktyvacijos funkcija")

# Generuojame tinkamus svorius ir bias
def find_valid_weights(activation):
    for _ in range(10000):  # Bandymų skaičius
        w = np.random.uniform(-1, 1, 2)
        b = np.random.uniform(-1, 1)
        output = neuron_activation(X, w, b, activation)
        if np.array_equal(output, labels):
            return w, b
    return None, None

# Randame tris tinkamus rinkinius su slenkstine funkcija
weights_bias_sets = [find_valid_weights('sigmoid') for _ in range(3)]


# Vizualizuojame duomenis, tieses ir svorių vektorius
plt.figure(figsize=(5, 5))
plt.scatter(class1[:, 0], class1[:, 1], color='green', label='Klasė 0')
plt.scatter(class2[:, 0], class2[:, 1], color='purple', label='Klasė 1')

colors = ['red', 'blue', 'orange']
for i, (w, b) in enumerate(weights_bias_sets):
    if w is not None:
        x_vals = np.linspace(0, 10, 100)
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, color=colors[i], linestyle='dashed', label=f'Tiesė {i+1}')
        
        # Pasirenkame tašką ant tiesės (x=5)
        x0 = 5  
        y0 = -(w[0] * x0 + b) / w[1]  
        
        # Vektorius iš šio taško, atitinkantis normalų vektorių
        plt.quiver(x0, y0, w[0], w[1], color=colors[i], angles='xy', scale_units='xy', scale=0.08, width=0.005)

plt.xlabel("X ašis")
plt.ylabel("Y ašis")
plt.title("Duomenys su klasifikavimo tiesėmis ir vektoriais")
plt.legend()
plt.grid(True)
plt.axis('equal')  # Užtikriname, kad vektoriai vizualiai būtų statmeni

# Nustatome x ir y ašių ribas nuo 0 iki 10
plt.xlim(0, 10)
plt.ylim(0, 10)

plt.show()
