import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import Model, RealData, ODR
import math

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

# aggiungo dati al progetto Python

URL = '/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/integratore.xlsx'

file = pd.read_excel(URL)

f = file["f"].values
vin = file["vin"].values
vout = file["vout"].values
fs_vin = file["fs vin"].values
fs_vout = file["fs vout"].values

Int = [f, vin ,vout, fs_vin, fs_vout]

# valore e incertezza della resistenze

R1 = 1800 # Ohm
errR1 = 30

R2 = 12000
errR2 = 30

# valore e incertezza del condensatore

C = 33 * 10**(-9)
errC = 3.3 * 10**(-9)

# erroe sulla frequenza

errFreq = Int[0] * 0.05

# errore del V{in}

errVin = Int[3] * 0.03

# errore del V_{out}

errVout = Int[4] * 0.03

#  errore per l'ascissa dei grafici

errXgrafico = np.abs( Int[0]*0.05 / (Int[0] * np.log(10) ) )

# definisco funzione per calcolare errore per l'ordinata dei grafici e li metto in una lista pre-dichiarata per evitare sbattimenti di 2 palle

def CalcoloErrYgrafico(v_in, errore_vin, v_out, errore_vout):
    return 20 / np.log(10) * np.sqrt( ( errore_vout / v_out )**2 + ( errore_vout / v_in )**2 )

errYgrafico = CalcoloErrYgrafico(Int[1], errVin, Int[2], errVout)

df1 = pd.DataFrame({
    r"$f$":Int[0],
    r"Inc $f$": errFreq,
    r"$V_{IN}$": Int[1],
    r"Inc $V_{IN}$": errVin,
    r"$V_{OUT}$": Int[2],
    r"Inc $V_{OUT}$": errVout
})
print(fr"{R2} $\Omega$")
print(df1.to_string(index=False))

# definisco funzione per calcolare i valori di y nel grafico 

def dB_pot(vout, vin):     
    return 20 * np.log10(vout / vin) 

xGrafico = np.log10(Int[0])
yGrafico = dB_pot(Int[2], Int[1])

plt.errorbar(xGrafico, yGrafico , xerr = errXgrafico, yerr = errYgrafico, fmt = "o", markersize=1, elinewidth=1, capsize=2, label="Dati integratore", color="coral")
plt.title(fr"Grafico Frequenza-Guadagno per $R_1$ = {R1} $\Omega$ e $R_2$ = {R2} $\Omega$" )
plt.xlabel(r"$log_{10}(\mathrm{f}) \ (decade)$" )
plt.ylabel(r"$20 \ log_{10}\left(\frac{V_{OUT}}{V_{IN}}\right) (dB)$" )
plt.legend()
plt.savefig(f"/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInt/graficoDati.png")
plt.show()


# calcolo valori teorici del guadagno

Ateo = R2 / R1
incAteo = np.sqrt( (errR2/R1)**2 + (R2 / R1**2 * errR1 )**2 )

# stampo valori del gudagno teorico

print(rf"Il valore del guadagno teorico per $R_1$ = {R1} $\Omega$$ e R_2$ = {R2} $\Omega$ è: ",Ateo, rf"$\pm$" ,incAteo)


# definisco definizioni per media pesata

def mediaPesata(x, sx):
    w = 1/sx**2
    return np.sum(w*x)/np.sum(w)

def incMediaPesata(sx):
    w = 1/sx**2
    return np.sqrt(1/np.sum(w))

# faccio la media pesata dei primi 10 elementi del set

Alab = mediaPesata(Int[2][:4] / Int[1][:4], np.sqrt( ( errVout[:4] / Int[1][:4] )**2 + ( - Int[2][:4] / Int[1][:4]**2 * errVin[:4] )**2  ))
incAlab = incMediaPesata(np.sqrt( ( errVout[:4] / Int[1][:4] )**2 + ( - Int[2][:4] / Int[1][:4]**2 * errVin[:4] )**2  ))

print(rf"Il valore del guadagno sperimentale per $R_1$ = {R1} $\Omega$$ e $ R_2$ = {R2} $\Omega$ è: ", Alab, rf"$\pm$" ,incAlab)

# Calcolo discepanza con la relativa incertezza

delta = np.abs( Alab - Ateo )
sigmaDelta = np.sqrt(incAlab**2 + incAteo**2)

print(rf"$\Delta$ = {delta} $\pm$ {sigmaDelta}")

# Calcolo valore t

t_val = delta / sigmaDelta

print("t =", t_val)

# definisco una funzione che vada a fare l'ODR dei dati dopo la frequenza di taglio

def FitRettaObliqua(x, y):
    modello = Model(lambda B, x: B[0] * x + B[1])  # Modello lineare
    dati = RealData(x, y)
    odr = ODR(dati, modello, beta0=[0, np.mean(y)])
    output = odr.run()
    return output.beta, output.sd_beta

# cerco parametri delle retta obliqua

Fit = FitRettaObliqua(xGrafico[23:28], yGrafico[23:28])

# stampo valori del coefficiente angolare della retta obliqua

print(rf"Il coefficiente angolare $\rho$della retta obliqua per $R_1$ = {R1} $\Omega$ e $R_2$ = {R2} $\Omega$ è pari a {Fit[0][0]} $\pm$ {Fit[1][0]} ")


# definisco la funzione che mi trovifrequenza di taglio

def FrequenzaTaglio(Alab, a, q):
    return 10**( ( 20 * np.log10(Alab) - q) / a )

# definisco la funzione per calcolare l'incertezza sulla frequenza di taglio - DA RIVEDERE

def incFrequenzaTaglio(A, a, q, incAlab, inca, incq):
    logA = np.log10(A)  # Logaritmo base 10 di A
    power_term_a = 20 * logA / a  # Calcolo di (20 * log10(A) / a)
    
    rispettoa = -2 * A**power_term_a * 10**(q / a) * np.log(10) * q
    rispettoq = - np.log(10) / a * A**power_term_a / 10**(q / a)
    rispettoA = 20 / (a * A) * A**power_term_a / 10**(q / a)
    
    return np.sqrt((rispettoa * inca)**2 + (rispettoA * incAlab)**2 + (rispettoq * incq)**2)


# calcolo frequenze di taglio

FreqTaglio = FrequenzaTaglio(Alab, Fit[0][0], Fit[0][1] )

# calcolo incertezze nelle frequenze di taglio

incFreqTaglio = incFrequenzaTaglio(Alab, Fit[0][0], Fit[0][1], incAlab, Fit[1][0], Fit[1][1] )

# calcolo frequenza di taglio teorica

FreqTaglioTeo = 1 / (2 * np.pi * R2 * C)

# calcolo incertezza nella frequenza di taglio teorica

df_dR2 = -1 / (2 * np.pi * R2**2 * C)

df_dC = -1 / (2 * np.pi * R2 * C**2)

incFreqTaglioTeo = np.sqrt((df_dR2 * errR2)**2 + (df_dC * errC)**2)

# scarto della frequenza di taglio

DiffFreq = np.abs(FreqTaglio - FreqTaglioTeo)
incDiffFreq = np.sqrt(incFreqTaglio**2 + incFreqTaglioTeo**2)

# stampa

print( rf"La frequenza di taglio sperimentale per per $R_1$ = {R1} $\Omega$ e $R_2$ = {R2} $\Omega$ è: ", FreqTaglio, r"$\pm$" , incFreqTaglio )

print( rf"La frequenza di taglio teorica per per $R_1$ = {R1} $\Omega$ e $R_2$ = {R2} $\Omega$ è: ", FreqTaglioTeo, r"$\pm$" , incFreqTaglioTeo )

print("Scarto tra frequenza di taglio e frequenza di taglio teorica:", DiffFreq, r"$\pm$", incDiffFreq)

# Stampo stampabilmente tutto lo stampabile

plt.errorbar(xGrafico,   yGrafico , xerr = errXgrafico, yerr = errYgrafico, fmt = "o", markersize=3, elinewidth=1, capsize=3, label="Dati sperimentali", color="coral")
#costante 
plt.plot( np.linspace(xGrafico[0], np.log10(FreqTaglio) + 0.5, 500), np.full( 500, dB_pot(Alab, 1) ),  label="Fit retta costante", color="coral", alpha=0.4)
#obliquo
x = np.linspace(np.log10(FreqTaglio) -0.5, xGrafico[-1], 500)
plt.plot( x, Fit[0][0]*x + Fit[0][1] , label="Fit retta obliqua", color="coral", alpha=0.4)

plt.title(fr"Grafico Frequenza-Guadagno per $R_1$ = {R1} $\Omega$ e $R_2 =$ {R2} $\Omega$" )
plt.xlabel(r"$log_{10}(\mathrm{f}) \ (decade)$" )
plt.ylabel(r"$20 \ log_{10}\left(\frac{V_{OUT}}{V_{IN}}\right) (dB)$" )
plt.legend()
plt.savefig(f"/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInt/graficoConFit.png")
plt.show()

# stampo differenza scarti frequenza

x = [0,1]
metodo = ["Teorico", "Sperimentale"]
plt.title("Scarti Frequenza di taglio")
plt.errorbar(x, [FreqTaglioTeo, FreqTaglio], yerr=[incFreqTaglioTeo, incFreqTaglio], fmt="o", color="coral", capsize=5)
plt.xticks(x, metodo)
plt.xlabel("Metodo")
plt.ylabel(r"$\nu (Hz)$")
plt.savefig("/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInt/graficoFreqTaglio.png")
plt.show()