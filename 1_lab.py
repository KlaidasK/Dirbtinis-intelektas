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
X = np.vstack((class1, class2))

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
for (x, y), label in zip(X, labels):
    print(f"Koordinatės: ({x:.2f}, {y:.2f}), Klasė: {label}")
    
# Aktyvacijos funkcija
activation = 'threshold' # 'threshold' (slenkstinė) arba 'sigmoid' (sigmoidinė)
weights_bias_sets = []

for i in range(3):  # Trys tinkami rinkiniai
    for i in range(10000):  # Bandymų skaičius
        w = np.random.uniform(-1, 1, 2)
        b = np.random.uniform(-1, 1)
        a = np.dot(X, w) + b

        if activation == 'threshold':
            output = np.where(a >= 0, 1, 0)
        elif activation == 'sigmoid':
            output = np.round(1 / (1 + np.exp(-a)))
        else:
            raise ValueError("Nežinoma aktyvacijos funkcija")

        if np.array_equal(output, labels):
            weights_bias_sets.append((w, b))
            print(f"Rasti {activation} tinkami svoriai {len(weights_bias_sets)}: {w}, Bias: {b}")
            break



# Vizualizuojame duomenis, tieses ir svorių vektorius
plt.figure(figsize=(5, 5))
plt.scatter(class1[:, 0], class1[:, 1], color='green', label='Klasė 0')
plt.scatter(class2[:, 0], class2[:, 1], color='purple', label='Klasė 1')

colors = ['red', 'blue', 'orange'] # Tiesių ir vektorių spalvos
for i, (w, b) in enumerate(weights_bias_sets):  
    if w is not None:   # Tikriname ar svoriai rasti
        x_vals = np.linspace(0, 10, 100) 
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, color=colors[i], linestyle='dashed', label=f'Tiesė {i+1}')
        
        # Pasirenkame tašką ant tiesės (x=5)
        x0 = 5  
        y0 = -(w[0] * x0 + b) / w[1]  
        
        # Vektoriai atvaizduojami kaip rodyklės
        plt.quiver(x0, y0, w[0], w[1], color=colors[i], angles='xy', scale_units='xy', scale=0.08, width=0.005, label=f'Vektorius {i+1}')

plt.xlabel("X ašis")
plt.ylabel("Y ašis")
plt.title("Duomenys su klasifikavimo tiesėmis ir vektoriais")
plt.legend()
plt.grid(True)
plt.axis('equal')  # Vektoriai statmeni

# Nustatome x ir y ašių ribas nuo 0 iki 10
plt.xlim(0, 10)
plt.ylim(0, 10)

plt.show()
