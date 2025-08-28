# Daugiasluoksnis Perceptronas – KNIME

Šis projektas skirtas analizuoti daugiasluoksnio perceptrono (MLP) veikimą naudojant KNIME platformą, atliekant klasifikavimo užduotis su **Iris** duomenų rinkiniu.

## Turinys

- **`iris.csv`** – pilnas duomenų rinkinys su 150 įrašų ir 5 stulpeliais: Sepalo ilgis, Sepalo plotis, Žiedlapio ilgis, Žiedlapio plotis ir Klasių etiketė.
- **`iris_train.csv`** – treniravimo duomenų rinkinys.
- **`iris_test.csv`** – testavimo duomenų rinkinys.
- **`LD3_Klaidas_Kubilius.pdf`** – laboratorinio darbo ataskaita su užduoties aprašymu, rezultatais ir išvadomis.

## Tikslai

- Sukurti daugiasluoksnį perceptroną naudojant KNIME.
- Mokyti modelį klasifikuoti Iris duomenis į tris klases.
- Analizuoti modelio tikslumą ir klaidas.
- Vizualizuoti mokymo procesą bei modelio rezultatus.

## Naudojami duomenys

**Iris duomenų rinkinys** yra klasikinis daugiaklasis klasifikavimo uždavinys, kuriame pateikiama 150 įrašų, suskirstytų į tris klases:

- **Setosa**
- **Versicolor**
- **Virginica**

Kiekvienas įrašas turi šiuos požymius:

- Sepalo ilgis (cm)
- Sepalo plotis (cm)
- Žiedlapio ilgis (cm)
- Žiedlapio plotis (cm)

## Darbo eiga

1. **Duomenų paruošimas**: Įkeliamas duomenų rinkinys ir suskirstomas į treniravimo bei testavimo duomenis.
2. **Modelio kūrimas**: KNIME aplinkoje sukuriamas daugiasluoksnis perceptronas su įėjimo, paslėptų sluoksnių ir išėjimo mazgais.
3. **Mokymas**: Modelis mokomas naudojant treniravimo duomenis.
4. **Vertinimas**: Modelio tikslumas vertinamas naudojant testavimo duomenis.
5. **Vizualizacija**: Generuojami grafikai, rodantys tikslumą ir klaidų analizę.

## Rezultatai

- Modelis sėkmingai klasifikavo duomenis į tris klases.
- Ataskaitoje pateikiama tikslumo, klaidų analizės bei vizualizacijos.

## Reikalavimai

- **KNIME Analytics Platform** – atvirojo kodo duomenų analizės platforma.
- 
## Nuorodos

- [Pilnas duomenų rinkinys (iris.csv)](https://github.com/KlaidasK/Dirbtinis-intelektas/blob/Daugiasluoksnis-perceptronas-KNIME/iris.csv)
- [Treniravimo duomenys (iris_train.csv)](https://github.com/KlaidasK/Dirbtinis-intelektas/blob/Daugiasluoksnis-perceptronas-KNIME/iris_train.csv)
- [Testavimo duomenys (iris_test.csv)](https://github.com/KlaidasK/Dirbtinis-intelektas/blob/Daugiasluoksnis-perceptronas-KNIME/iris_test.csv)
- [Laboratorinio darbo ataskaita (LD3_Klaidas_Kubilius.pdf)](https://github.com/KlaidasK/Dirbtinis-intelektas/blob/Daugiasluoksnis-perceptronas-KNIME/LD3_Klaidas_Kubilius.pdf)
