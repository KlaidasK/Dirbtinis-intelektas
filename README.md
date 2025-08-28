# Dirbtinis Neuronas su Klasifikavimo Užduotimis

Šis projektas skirtas išanalizuoti paprasčiausio dirbtinio neurono veikimo principą, apmokyti neuroną klasifikuoti.


## Tikslas

- Paruošti ir sumaišyti duomenų rinkinį su dviem klasėmis (vėžio diagnozė: 0 – benign, 1 – malign).  
- Sukurti paprastą dirbtinį neuroną su sigmoidine aktyvacijos funkcija.  
- Išbandyti įvairius mokymosi metodus: batch ir stochastic gradient descent.  
- Vizualizuoti mokymo/validacijos paklaidas ir tikslumus bei testavimo rezultatus.  
- Nustatyti optimalų mokymosi greitį (learning rate).

## Projekto Turinys

- 2_lab.py – pagrindinis skriptas: modelio mokymas, testavimas, vizualizacija  
- 2_lab_data.py – duomenų paruošimo skriptas  
- cleaned_breast_cancer.csv – paruoštas duomenų rinkinys  
- breast-cancer-wisconsin.data – originalus duomenų rinkinys   
- LD2_Klaidas_Kubilius.pdf – laboratorinio darbo ataskaita

## Duomenų Paruošimas su 2_lab_data.py

Naudojamas Breast Cancer Wisconsin duomenų rinkinys (breast-cancer-wisconsin.data).  

Veikimo principas:

1. Įkeliami duomenys su stulpeliais:
   Clump_Thickness, Uniformity_Cell_Size, Uniformity_Cell_Shape, Marginal_Adhesion, Single_Epithelial_Cell_Size, Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses, Class

2. Pašalinamas ID stulpelis  
3. Klaustukai (?) pakeičiami į NaN, eilutės su trūkstamomis reikšmėmis pašalinamos  
4. Class reikšmės konvertuojamos: 2 → 0 (benign), 4 → 1 (malign)  
5. Duomenys atsitiktinai permaišomi (shuffle) su nustatytu seed (random_state=100)  
6. Išvalyti duomenys išsaugomi kaip cleaned_breast_cancer.csv  


## Dirbtinio Neurono Modelis

- Aktyvacijos funkcija: Sigmoidinė

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

- Gradientinis nusileidimas (Gradient Descent)  
  - Batch – gradientai skaičiuojami visam mokymo rinkiniui  
  - Stochastic (SGD) – gradientai skaičiuojami po vieną pavyzdį  

- Klasifikacija: Prognozė >= 0.5 → klasė 1, kitaip klasė 0


## Modelio Mokymas

- Parametrai:
  - Mokymosi greitis (learning_rate)  
  - Epochos (epochs)  
  - Mokymo metodas (batch arba stochastic)  

- Saugojami mokymo ir validacijos paklaidos bei tikslumo istorijos  
- Išvedami rezultatai kas 100 epochų


## Testavimas

- Duomenys padalijami į mokymo, validacijos ir testavimo rinkinius  
- Testavimo tikslumas ir paklaida apskaičiuojami paskutinėje epochoje  
- Išvedamos kiekvieno testavimo įrašo prognozės  
- Nustatomas optimalus mokymosi greitis pagal validacijos tikslumą


## Vizualizacijos

- Paklaidos ir tikslumo grafikai:
  - Mokymo vs validacijos paklaida  
  - Mokymo vs validacijos tikslumas  

- Stulpelinė diagrama: testavimo tikslumas pagal skirtingus mokymosi greičius

## Naudojimo Instrukcijos

1. Įdiekite reikalavimus:

pip install numpy pandas matplotlib scikit-learn

2. Paruoškite duomenis:

python 2_lab_data.py

3. Paleiskite pagrindinį skriptą:

python 2_lab.py

4. Peržiūrėkite rezultatus ir grafikus

## Licencija

Šis projektas yra paskelbtas pagal GPL-3.0 licenciją
