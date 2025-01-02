import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import math
from scipy.odr import Model, RealData, ODR
from scipy.stats import norm
import time

# imposto parametri di stampa 
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "font.size": 12,  # font
    "axes.titlesize": 14,  # titolo degli assi
    "axes.labelsize": 12,  # etichette degli assi
    "xtick.labelsize": 10,  # etichette degli assi x
    "ytick.labelsize": 10,  # etichette degli assi y
})

# lunghezza pendolo
lungPend = 0.579 # mm
incLungPend = 0.001 # mm

# metto variabili globali per tutti i file
inc = 0.005

# importo dati
for i in range(1, 13):
    try:
        file = pd.read_excel(f'/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione4/Dati/NewFile{i}.xlsx')
        X = file["X"].values * inc
        Y = file["Volt"].values / 10 # si è diviso per dieci a causa di un moltiplicatore nell'oscilloscopio
        
        if i == 1:
            f1 = np.array([
            #   0  1
                X, Y
            ])
        elif i == 2:
            f2 = np.array([
                X, Y
            ])
        elif i == 3:
            f3 = np.array([
                X, Y
            ])
        elif i == 4:
            f4 = np.array([
                X, Y
            ])
        elif i == 5:
            f5 = np.array([
                X, Y
            ])
        elif i == 6:
            f6 = np.array([
                X, Y
            ])
        elif i == 7:
            f7 = np.array([
                X, Y
            ])
        elif i == 8:
            f8 = np.array([
                X, Y
            ])
        elif i == 9:
            f9 = np.array([
                X, Y
            ])
        elif i == 10:
            f10 = np.array([
                X, Y
            ])

    except KeyError as e:
        print(f"Errore nel file {i}: {e}")
    except Exception as e:
        print(f"Errore generico nel file {i}: {e}")
    
#file di tutto
file = np.array([
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
])

# print
for i in range(len(file)):
    plt.title(f"Dati raccolti della serie {i+1}")
    plt.errorbar(file[i][0], file[i][1], fmt="o", color="royalblue", markersize=2, elinewidth=1, capsize=1, label="Dati")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Potenziale (V)")
    plt.legend()
    #plt.show(block=False)
    plt.savefig(f'/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione4/Grafici/Dati{i+1}.png')
    plt.draw()
    plt.pause(0.5)
    plt.close()

# imposto valore di soglia
ValSoglia = 4

# primo indice - indice del valore della seconda buca più vicino al valore soglia (andamento decrescente)
def primoSoglia(x):
    for i in range(1, len(x[1])):
        if x[1][i] < ValSoglia and x[1][i-1] > ValSoglia:
            k = i
            break
    for j in range(k+1,len(x[1])):
        if x[1][j] < ValSoglia and x[1][j-1] > ValSoglia:
            return j if ValSoglia - x[1][j] < x[1][j-1] - ValSoglia else j-1

#ultimo indice - indice del valore dell'ultima buca più vicino al valore soglia (andamento decrescente)
def ultimoSoglia(x):
    for i in range(1, len(x[1])):
        if x[1][-i]<ValSoglia and x[1][-i-1]>ValSoglia:
            return len(x[1])-i if ValSoglia - x[1][-i] < x[1][-i-1] - ValSoglia else len(x[1])-i-1

# periodo
def Periodo(x):
    primo = primoSoglia(x)
    ultimo = ultimoSoglia(x)
    print(primo, ultimo)
    
    # stampa con indici
    plt.title(fr"Dati raccolti della serie {i+1} con indici $t_0$ e $t_1$")
    plt.errorbar(x[0], x[1], fmt="o", color="royalblue", markersize=2, elinewidth=1, capsize=1, label="Dati")
    plt.axvline(x[0][primo], color="pink", label=r"$t_0$")
    plt.axvline(x[0][ultimo], color="orange", label=r"$t_1$")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Potenziale (V)")
    plt.legend()
    plt.savefig(f'/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione4/Grafici/Indici{i+1}.png')
    plt.draw()
    plt.pause(0.5)
    plt.close()
    if primo is not None and ultimo is not None:
        print(x[0][ultimo], x[0][primo])
        return (x[0][ultimo] - x[0][primo]) / 3

# incertezza sul periodo
def incPeriodo():
    return np.sqrt(2 * ( inc / 2 / np.sqrt(12)) ** 2) / 3

# Calc periodi
periodi = []
incPeriodi = []

for i in range(len(file)):
    T = Periodo(file[i])
    
    if T is not None:
        periodi.append(T)
        incPeriodi.append(incPeriodo())
        print(f"Periodo N.{i+1}: {periodi[i]:.4f} ± {incPeriodi[i]:.4f} s")
    else:
        print(f"Periodo N.{i+1}: Non calcolabile")

def media(x):
    return np.mean(x)

def incPeriodoMedio(p):
    T = np.mean(p)
    return np.sqrt(np.sum((p - T)**2)/9) / np.sqrt(10)

# media periodi
mPeriodi = np.mean(periodi)
incmPeriodi = incPeriodoMedio(periodi)

print(fr"Periodo medio: {mPeriodi} $\pm$ {incmPeriodi}")

# istogramma - Gauss
plt.hist(periodi, bins=[1.2, 1.48, 1.51, 1.55, 1.7], density=True, alpha=0.6, color='skyblue', label='Dati osservati', range=(1.2, 1.7))

# Creazione del fit gaussiano
x = np.linspace(min(periodi) -1, max(periodi)+1, 1000)
gauss_fit = norm.pdf(x, mPeriodi, incmPeriodi)  # Calcola la distribuzione gaussiana

# Tracciare la curva del fit
plt.plot(x, gauss_fit, 'r-', label=rf'Fit Gaussiano $\mu$={mPeriodi:.2f}, $\sigma$={incmPeriodi:.2f}')

# Etichette e legenda
plt.title("Distribuzione dei periodi con fit gaussiano")
plt.xlabel("Periodo (s)")
plt.ylabel("Densità di probabilità")
plt.xlim((1.2,1.7))
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione4/Grafici/DistribuzioneDensita.png')
plt.draw()
plt.pause(0.5)
plt.close()

# istogramma - Gauss
plt.hist(periodi, bins=20 , alpha=0.6, color='skyblue', label='Dati osservati', range=(1.2, 1.7))
# Tracciare la curva del fit
plt.plot(x, gauss_fit, 'r-', label=rf'Fit Gaussiano $\mu$={mPeriodi:.2f}, $\sigma$={incmPeriodi:.2f}')
# Etichette e legenda
plt.title("Distribuzione dei periodi con fit gaussiano")
plt.xlabel("Periodo (s)")
plt.ylabel("Ricorrenza")
plt.xlim((1.2,1.7))
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione4/Grafici/Distribuzione.png')
plt.draw()
plt.pause(0.5)
plt.close()

# verosimiglianza
mu_values = np.linspace(1.2, 1.7, 1000)
log_likelihood = [
    np.sum(norm.logpdf(periodi, loc=mu, scale=incmPeriodi)) for mu in mu_values
]

plt.hist(periodi, bins=20, density=False, alpha=0.6, color='skyblue', label='Dati osservati', range=(1.2, 1.7))

# Disegno della funzione di log-verosimiglianza
plt.plot(mu_values, np.exp(log_likelihood - np.max(log_likelihood)), 'r-', label='Likelihood (verosimiglianza normalizzata)')

# Etichette e legenda
plt.title("Distribuzione dei periodi con likelihood")
plt.xlabel("Periodo (s)")
plt.ylabel("Densità di probabilità / Likelihood")
plt.xlim((1.2, 1.7))
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione4/Grafici/DistribuzioneVerosimiglianza.png')
plt.draw()
plt.pause(0.5)
plt.close()

# calcolo g con incertezza 
g =  4*np.pi**2*lungPend/mPeriodi**2
incg = np.sqrt( (4*np.pi**2/mPeriodi**2 * incLungPend)**2 + ( -8*np.pi**2*lungPend/mPeriodi**3 * incmPeriodi )**2 )

print(rf"L'accelerazione di gravità calcolata è {g:.3f} $\pm$ {incg:.3f}")