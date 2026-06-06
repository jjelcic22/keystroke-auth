# Identificiranje korisnika pomoću dinamike tipkanja
> Projekt iz kolegija **Sigurnost Interneta**

Autori:
- Jure Jelčić
- Mateo Čuvalo
- Roberto Šandro
- Maksim Kos


## O projektu

Sustav za kontinuiranu autentifikaciju korisnika na temelju dinamike tipkanja. Svaki korisnik ima svoj prepoznatljiv način tipkanja, pa se na temelju raznih čimbenika može provjeriti je li za tipkovnicom stvarno prijavljeni korisnik ili netko drugi

Za dva autorizirana korisnika (Jure i Mateo), istrenirani su zasebni binarni modeli koji razlikuju te korisnike od ostalih

**Kako radi:**
1. Korisnik se prijavi kao Jure ili Mateo
2. Počinje tipkati u tekstualno polje, a frontend bilježi vrijeme svakog pritiska i otuštanja gumba
3. Svakih 40 pritiska, šalje se jedan prozor na backend, koji iz tih podataka računa značajke (vrijeme držanja tipke, razmake između pritiska, brzinu tipkanja i sl.) i šalje ih modelu prijavljenog korisnika
4. Model vraća vjerojatnost da je riječ o pravom korisniku, te ako je ona iznad praga, prozor je prihvaćen
5. Na temelju zadnjih 5 prozora određuje se status sesije:
   - 0 ili 1 loš prozor -> **authenticated**
   - 2 loša prozora -> **warning**
   - 3 ili više loša prozora -> **locked**


## Struktura repozitorija
```
keystroke-auth/
├── app.py                      # Flask backend
├── requirements.txt            # Python ovisnosti
│
├── data/                       # Mapa s podacima
│   ├── raw_samples.json        # Spojeni sirovi uzorci tipkanja
│   ├── features.csv            # Izračunate značajke
│   └── raw_inputs.zip          # Arhiva originalnih prikupljenih datoteka
│
├── ml/                         
│   ├── prepare_dataset.py      # Spaja sirove uzorke i generira features.csv
│   ├── features.py             # Izračun značajki iz prozora tipkanja
│   ├── model_pipeline.ipynb    # Treniranje, evaluacija i spremanje modela
│   └── predict.py              # Učitavanje modela i predikcija za prijavljenog korisnika
│
├── models/                     # Istrenirani modeli, scaler i prag odličivanja za korisnike
│   ├── jure_model.pkl           
│   ├── jure_scaler.pkl         
│   ├── jure_threshold.json     
│   ├── mateo_model.pkl         
│   ├── mateo_scaler.pkl        
│   ├── mateo_threshold.json    
│   └── feature_columns.json    # Redoslijed značajki koje model očekuje
│
├── templates/                  # HTML stranice
│   ├── index.html              # Glavna stranica za live autentifikaciju
│   └── collector.html          # Alat za prikupljanje uzoraka tipkanja
│
└── static/                     # Frontend resursi
    ├── script.js               # Logika hvatanja tipki i komunikacija s backendom
    └── style.css               # Dizajn za frontend
```

## Pokretanje projekta

### 1. Kloniranje repozitorija
```bash
git clone <https://github.com/jjelcic22/keystroke-auth.git>
cd keystroke-auth
```

### 2. Izrada virtualnog okruženja
```bash
python3 -m venv .venv
```

Aktivacija:
```bash
source .venv/bin/activate
```

### 3. Instalacija ovisnosti
```bash
pip install -r requirements.txt
```

### 4. Pokretanje aplikacije
```bash
python3 app.py
```

Aplikacija se pokreće na adresi:
```
http://127.0.0.1:5000/
```

Nakon što se aplikacija otvori u pregledniku, potrebno je odabrati korisnika i krenuti tipkati


## Prikupljanje novih uzoraka

Datoteka `templates/collector.html` je samostalan alat za prikupljanje uzoraka
tipkanja i ne treba mu backend
- Otvara se u pregledniku, odabere se osoba koja tipka, te se na kraju spremaju svi uzorci u *.json* formatu 


## Ponovno treniranje modela

Ako se prikupe novi uzorci, model se može iznova istrenirati:
1. Generiranje skupa značajki iz sirovih uzoraka:
```bash
python3 ml/prepare_dataset.py --input data/raw_samples.json
```
- Ova naredba stvara `data/raw_samples.json` i `data/features.csv`

2. Otvaranje `ml/model_pipeline.ipynb` i pokretanje svih ćelija. Notebook trenira modele, evaluira ih i sprema nove `.pkl` datoteke u mapu `models/`

## Korištene tehnologije

- **Python** (Flask, scikit-learn, pandas, numpy, joblib)
- **RandomForestClassifier** za klasifikaciju
- **HTML / CSS / JavaScript** za frontend
