import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Sigmoidinė aktyvavimo funkcija
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoidinės funkcijos išvestinė
def sigmoid_derivative(x):
    return x * (1 - x)

# Funkcija, skirta modelio mokymui su batch ar stochastic metodu
def train_model(X_train, y_train, X_val, y_val, learning_rate, epochs, n_features, method='batch'):
    w = np.random.uniform(0, 1, (n_features, 1))
    b = 0.0

    # Laiko matavimas
    start_time = time.time()

    # Saugojimui
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    m = len(X_train)  # Mokymo duomenų dydis
    epoch = 0

    while epoch < epochs:
        gradientSum = np.zeros(n_features) # Masyvas, kuriame saugomi gradientų sumos

        if method == 'batch':
            # Batch mokymo metodas
            z = np.dot(X_train, w) + b # Funkcija Xw + b
            y_pred = sigmoid(z) # Priskiria arčiausiai 0 arba 1
            error = y_pred - y_train # Netikslumas tarp prognozės ir tikros reikšmės
            gradient_w = np.dot(X_train.T, error) / len(X_train) # Svorių gradientas
            gradient_b = np.mean(error)    # Poslinkio gradientas
            
            w -= learning_rate * gradient_w # Atnaujina svorius
            b -= learning_rate * gradient_b # Atnaujina poslinkį

        elif method == 'stochastic':
            # Stochastic mokymo metodas
            X_train, y_train = shuffle(X_train, y_train) # Duomenų maišymas
            for i in range(len(X_train)):
                xi = X_train[i].reshape(1, -1) # Paimama i-oji treniravimo duomenų eilutė ir paverčiama 2D eilutės vektoriumi
                yi = y_train[i].reshape(1, -1) # Paimama i-oji tikroji reikšmė ir paverčiama 2D eilutės vektoriumi
                linear_output = np.dot(xi, w) + b # Funkcija Xw + b
                y_pred = sigmoid(linear_output) # Priskiria arčiausiai 0 arba 1
                error = yi - y_pred # Netikslumas tarp prognozės ir tikros reikšmės
                w_gradient = np.dot(xi.T, error * sigmoid_derivative(y_pred))  # Svorių gradientas
                b_gradient = np.sum(error * sigmoid_derivative(y_pred)) # Poslinkio gradientas
                
                # Svorių ir poslinkio atnaujinimas
                w += learning_rate * w_gradient
                b += learning_rate * b_gradient

        # Klasifikavimo tikslumo ir paklaidos skaičiavimas
        train_output = sigmoid(np.dot(X_train, w) + b)  # Apskaičiuojama prognozė mokymo duomenims
        train_predictions = (train_output >= 0.5).astype(int)
        train_accuracy = np.mean(train_predictions == y_train)
        train_loss = np.mean(np.abs(y_train - train_output))

        val_output = sigmoid(np.dot(X_val, w) + b)  # Apskaičiuojama prognozė validacijos duomenims
        val_predictions = (val_output >= 0.5).astype(int)
        val_accuracy = np.mean(val_predictions == y_val)
        val_loss = np.mean(np.abs(y_val - val_output))

        # Išsaugoti paklaidas ir tikslumus
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        epoch += 1

        # Kiekvienos 100 epochų metu išvedami rezultatai
        if epoch % 100 == 0:
            print(f"Epochos {epoch}: Mokymo paklaida {train_loss:.4f}, Mokymo tikslumas {train_accuracy * 100:.2f}%")
            print(f"Epochos {epoch}: Validacijos paklaida {val_loss:.4f}, Validacijos tikslumas {val_accuracy * 100:.2f}%")
    
    # Laiko skaičiavimas
    end_time = time.time()
    training_time = end_time - start_time

    return w, b, train_losses, val_losses, train_accuracies, val_accuracies, training_time


# Duomenų įkėlimas ir paruošimas
file_path = "cleaned_breast_cancer.csv"
df = pd.read_csv(file_path)
X = df.drop(columns=["Class"]).values
y = df["Class"].values.reshape(-1, 1)

# Duomenų padalijimas į mokymo, validacijos ir testavimo rinkinius
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=100)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=100)

# Modelio parametrai
learning_rate = 0.1
epochs = 1000
n_features = X_train.shape[1] #požymiai
method = 'stochastic'  # Pasirinkti mokymo metodą: batch arba stochastic

# Mokymas
w, b, train_losses, val_losses, train_accuracies, val_accuracies, training_time = train_model(
    X_train, y_train, X_val, y_val, learning_rate, epochs, n_features, method)

# Testavimas
test_output = sigmoid(np.dot(X_test, w) + b)
test_predictions = (test_output >= 0.5).astype(int)
test_accuracy = np.mean(test_predictions == y_test)
test_loss = np.mean(np.abs(y_test - test_output))

# Rezultatų išvedimas
print("Galutiniai svoriai:", w.flatten())
print("Galutinis poslinkis:", b)
print(f"Testavimo paklaida: {test_loss:.4f}")
print(f"Testavimo tikslumas: {test_accuracy * 100:.2f}%")
print(f"Mokymo laikas: {training_time:.2f} sekundes")

# Grafikai
plt.figure(figsize=(12, 5))

# Paklaidos grafikas
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label="Mokymo paklaida")
plt.plot(range(epochs), val_losses, label="Validacijos paklaida")
plt.xlabel("Epochos")
plt.ylabel("Paklaida")
plt.title("Mokymo ir Validacijos Paklaida")
plt.legend()

# Tikslumo grafikas
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accuracies, label="Mokymo tikslumas")
plt.plot(range(epochs), val_accuracies, label="Validacijos tikslumas")
plt.xlabel("Epochos")
plt.ylabel("Tikslumas")
plt.title("Mokymo ir Validacijos Tikslumas")
plt.legend()

plt.show()

# Įvairūs mokymosi greičiai
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.4, 0.8]  # Skirtingi mokymosi greičiai
test_accur_lr = []

# Saugo validavimo tikslumus ir kitus reikalingus duomenis
val_accuracies = [] # Saugo validavimo tikslumus
train_losses_all = []  # Saugo visus mokymo praradimus
val_losses_all = []    # Saugo visus validavimo praradimus
train_accuracies_all = []  # Saugo visus mokymo tikslumus
val_accuracies_all = []    # Saugo visus validavimo tikslumus

for lr in learning_rates:
    # skirtingi mokymosi greičiai
    w, b, train_losses, val_losses, train_accuracies, val_accuracies_temp, training_time = train_model(
        X_train, y_train, X_val, y_val, lr, epochs, n_features, method)  

    # Išsaugome praradimus ir tikslumus visoms epochoms
    train_losses_all.append(train_losses)
    val_losses_all.append(val_losses)
    train_accuracies_all.append(train_accuracies)
    val_accuracies_all.append(val_accuracies_temp)

    # Išsaugome paskutinės epochos validavimo tikslumą
    val_accuracies.append(val_accuracies_temp[-1] if val_accuracies_temp else 0)

    # Testavimas
    test_output = sigmoid(np.dot(X_test, w) + b)
    test_predictions = (test_output >= 0.5).astype(int)
    test_accuracy = np.mean(test_predictions == y_test)
    test_accur_lr.append(test_accuracy)

# Surasti geriausią rezultatą pagal validavimo duomenis
best_idx = np.argmax(val_accuracies)  # Indeksas, kurio validavimo tikslumas didžiausias
best_lr = learning_rates[best_idx]  # Geriausias mokymosi greitis

# Gautos reikšmės
best_train_loss = train_losses_all[best_idx][-1]  # Paskutinės epochos mokymo paklaida
best_val_loss = val_losses_all[best_idx][-1]  # Paskutinės epochos validavimo paklaida
best_train_acc = train_accuracies_all[best_idx][-1]  # Paskutinės epochos mokymo tikslumas
best_val_acc = val_accuracies_all[best_idx][-1]  # Paskutinės epochos validavimo tikslumas

# Paklaida testavimo duomenims
test_output = sigmoid(np.dot(X_test, w) + b)
test_predictions = (test_output >= 0.5).astype(int)
test_loss = np.mean((test_predictions - y_test) ** 2) # Testavimo paklaida
test_accuracy = np.mean(test_predictions == y_test)  # Testavimo tikslumas

# Geriausi rezultatai
print(f"Geriausias mokymosi greitis: {best_lr}")
print(f"Epochu skaicius: {epochs}")
print(f"Mokymo paklaida paskutineje epochoje: {best_train_loss:.4f}")
print(f"Validavimo paklaida paskutineje epochoje: {best_val_loss:.4f}")
print(f"Mokymo tikslumas paskutineje epochoje: {best_train_acc:.4f}")
print(f"Validavimo tikslumas paskutineje epochoje: {best_val_acc:.4f}")
print(f"Testavimo paklaida: {test_loss:.4f}")
print(f"Testavimo tikslumas: {test_accuracy:.4f}")

# Kiekvieno testavimo įrašo prognozės
for i, (pred, actual) in enumerate(zip(test_predictions, y_test)):
    print(f"Testavimo irasas {i + 1}: Nustatyta klase: {pred}, Tikroji klase: {actual}")

#Stulpeline diagrama mokymosi greičiui
plt.figure(figsize=(8, 6))
bars = plt.bar([str(lr) for lr in learning_rates], test_accur_lr, color='blue')
plt.xlabel('Mokymosi Greitis')
plt.ylabel('Testavimo tikslumas')
plt.title('Testavimo tikslumas pagal mokymosi greiti')

for bar, accuracy in zip(bars, test_accur_lr):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval / 2, f'{accuracy:.2f}', ha='center', va='center', color='white')

plt.show()



