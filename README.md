# Dirbtinio neurono modelis

Šis projektas skirtas išanalizuoti paprasčiausio dirbtinio neurono veikimo principą, sugeneruoti duomenis, apmokyti neuroną juos atskirti ir pavaizduoti rezultatus grafiškai.

## Tikslas
- Sugeneruoti duomenų rinkinį su dviem klasėmis.
- Realizuoti paprastą dirbtinį neuroną naudojant slenkstinę ir sigmoidinę aktyvacijos funkcijas.
- Rasti tinkamus svorius ir poslinkį (bias), leidžiančius teisingai klasifikuoti duomenis.
- Vizualizuoti klases, atskyrimo tieses ir svorių vektorius.

## Turinys
- **`1_lab.py`** – pagrindinis Python skriptas:
  - Sugeneruoja duomenis.
  - Atlieka klasifikaciją su dirbtiniu neuronu.
  - Ieško tinkamų svorių ir poslinkio.
  - Vizualizuoja klasifikavimo ribas ir svorių vektorius.
- **`test.py`** – testavimo skriptas.
- **`linear_separable_data.csv`** – pavyzdinis duomenų rinkinys.
- **`LD1_Klaidas_Kubilius.pdf`** – laboratorinio darbo ataskaita su užduoties aprašymu, kodo fragmentais ir išvadomis.

## Programos veikimo principas
1. Sugeneruojami duomenų taškai dviem klasėms:
   - **Klasė 0** – koordinatės tarp 1 ir 5.
   - **Klasė 1** – koordinatės tarp 6 ir 10.
2. Atsitiktinai generuojami svoriai (**w1, w2**) ir poslinkis (**b**).
3. Taškai klasifikuojami pagal aktyvacijos funkciją:
- **Slenkstinė (threshold):**  
  Jei linijinė kombinacija  
  `a = w1*x1 + w2*x2 + b`  
  yra didesnė arba lygi 0, klasė priskiriama 1, kitu atveju – 0.

- **Sigmoidinė (sigmoid):**  
  Rezultatas apskaičiuojamas pagal formulę:  
  `f(a) = 1 / (1 + exp(-a))`  
  Kadangi rezultatas yra tarp 0 ir 1, klasė nustatoma apvalinant:  
  - Jei `f(a) >= 0.5`, klasė = 1  
  - Jei `f(a) < 0.5`, klasė = 0
  
4. Ieškoma tokių parametrų, kurie užtikrina teisingą klasifikaciją.
5. Nubraižomos atskyrimo tiesės ir svorių vektoriai.

## Rezultatai
- Abi aktyvacijos funkcijos (threshold ir sigmoid) davė vienodus rezultatus.
- Sugeneruotos klasifikavimo tiesės ir svorių vektoriai vizualiai parodo, kaip neuronas priima sprendimus.
- Modelis puikiai veikia su linijiškai atskiriamais duomenimis, bet **nebūtų tinkamas persidengiančioms klasėms**.
