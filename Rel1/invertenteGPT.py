import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import math
from scipy.odr import Model, RealData, ODR

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

# definisco la variabile $R_1 \Omega$

R2 = 10000 # \pm 3
incR2 = 300

# definisco i colori per i grafici
colori = ["coral", "coral", "coral", "coral"]

# lista con resistenze R2  - REMINDER: oscilloscopio

R1 = [20000, 30000, 40000, 50000] # \pm 0.3
incR1 = 300

# aggiungo dati al progetto Python

URL = '/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/dati.xlsx'
fogliDaCopiare = ["10-20", "10-30", "10-40", "10-50"]

i = 0

for foglio in fogliDaCopiare:
    file = pd.read_excel(URL, sheet_name=foglio, decimal='.')
    
    f = file["f"].values
    err_f = file["inc f"].values
    vin = file["vin"].values
    vout = file["vout"].values
    inc_vin = file["inc vin"].values
    inc_vout = file["inc vout"].values
    
    if i == 0:
        Inv20 = np.array([
        #   0   1     2      3        4        5
            f, vin, vout, inc_vin, inc_vout, err_f
        ])
    elif i == 1:
        Inv30 = np.array([
            f, vin, vout, inc_vin, inc_vout, err_f
        ])
    elif i == 2:
        Inv40 = np.array([
            f, vin, vout, inc_vin, inc_vout, err_f
        ])
    elif i == 3:
        Inv50 = np.array([
            f, vin, vout, inc_vin, inc_vout, err_f
        ])
    i += 1

# inserisco tutti i dati in una lista, cosi da poterla iterare
Inv = [Inv20, Inv30, Inv40, Inv50]  # Lista ( (30, 5) , 4)

# erroe sulla frequenza
errFreq = [inv[5] for inv in Inv]

# errore del V{in}
errVin = [inv[3] for inv in Inv]

# errore del V_{out}
errVout = [inv[4] for inv in Inv]

#  errore per l'ascissa dei grafici
errXgrafico = [np.abs( inv[5] / (inv[0] * np.log(10) ) ) for inv in Inv]

# definisco funzione per calcolare errore per l'ordinata dei grafici e li metto in una lista pre-dichiarata
errYgrafico = []

def CalcoloErrYgrafico(v_in, errore_vin, v_out, errore_vout):
    return 20 / np.log(10) * np.sqrt( ( errore_vout / v_out )**2 + ( errore_vout / v_in )**2 )

for i in range(len(Inv)):
    errYgrafico.append(CalcoloErrYgrafico(Inv[i][1], errVin[i], Inv[i][2], errVout[i]))

# stampo a schermo dataframe per visualizzare 
df = []

for i in range(len(Inv)):
    df_loc = pd.DataFrame({
        r"$f$":Inv[i][0],
        r"Inc $f$": errFreq[i],
        r"$V_{IN}$": Inv[i][1],
        r"Inc $V_{IN}$": errVin[i],
        r"$V_{OUT}$": Inv[i][2],
        r"Inc $V_{OUT}$": errVout[i]
    })
    print(fr"{R1[i]} $\Omega$")
    print(df_loc.to_string(index=False))
    df.append(df_loc)


# definisco funzione per calcolare i valori di y nel grafico 
def dB_pot(vout, vin):     
    return 20 * np.log10(vout / vin) 


# calcolo valori x y del grafico
xGrafico = [np.log10(inv[0]) for inv in Inv]
yGrafico = [dB_pot(inv[2], inv[1]) for inv in Inv]


for i in range(len(Inv)):
    plt.errorbar(xGrafico[i],   yGrafico[i] , xerr = errXgrafico[i], yerr = errYgrafico[i], fmt = "o", markersize=2, elinewidth=1, capsize=1, label="Dati invertente",color=colori[i])
    plt.title(fr"Grafico Frequenza-Guadagno per $R_1 =$ {R1[i]} $\Omega$" )
    plt.xlabel(r"$log_{10}(\mathrm{f}) \ (decade)$" )
    plt.ylabel(r"$20 \ log_{10}\left(\frac{V_{OUT}}{V_{IN}}\right) (dB)$" )
    plt.legend()
    plt.savefig(f"/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInv/{R1[i]}graficoDati.png")
    plt.show()

# calcolo valori teorici del guadagno
Ateo = [ R2/res for res in R1]
incAteo = [ np.sqrt( (incR2/res)**2 + (R2 / res**2 * incR1 )**2 ) for res in R1]

# stampo valori del gudagno teorico
print("\n")
for i, val in enumerate(Ateo):
    print(rf"Il valore del guadagno teorico per $R_1$ = {R1[i]} $\Omega$ è: ",Ateo[i], rf"$\pm$" ,incAteo[i])

# definisco definizioni per media pesata
def mediaPesata(x, sx):
    w = 1/sx**2
    return np.sum(w*x)/np.sum(w)

def incMediaPesata(sx):
    w = 1/sx**2
    return np.sqrt(1/np.sum(w))

# faccio la media pesata dei primi 10 elementi del set
Alab = [mediaPesata(inv[2][:5] / inv[1][:5], np.sqrt( ( errVout[i][:5] / inv[1][:5] )**2 + ( - inv[2][:5] / inv[1][:5]**2 * errVin[i][:5] )**2  )) for i, inv in enumerate(Inv)]
incAlab = [incMediaPesata(np.sqrt( ( errVout[i][:5] / inv[1][:5] )**2 + ( - inv[2][:5] / inv[1][:5]**2 * errVin[i][:5] )**2  )) for i, inv in enumerate(Inv)]
print("\n")
for i, val in enumerate(Ateo):
    print(rf"Il valore del guadagno sperimentale per $R_1$ = {R1[i]} $\Omega$ è: ", Alab[i], rf"$\pm$" ,incAlab[i])

# Calcolo discepanza con la relativa incertezza
delta = [ np.abs( Alab[i] - Ateo[i] ) for i in range(len(Ateo)) ]
sigmaDelta = [ np.sqrt(incAlab[i]**2 + incAteo[i]**2) for i in range(len(Alab))]

# Calcolo valore t
t_val = [ delta[i] / sigmaDelta[i] for i in range(len(delta)) ]

xval = [0, 1]
metodo = ["Teorico", "Sperimentale"]
for i in range(len(R1)):
    plt.errorbar(xval, [Ateo[i], Alab[i]], yerr= [incAteo[i], incAlab[i]], fmt="o", color=colori[i],capsize=5)
    plt.xticks(xval, metodo)
    #additivi
    plt.title(fr"Grafico confronto guadagni per $R_1 =$ {R1[i]} $\Omega$")
    plt.ylabel("A" )
    plt.xlabel("Metodo")
    plt.savefig(f"/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInv/{R1[i]}graficoConFit.png")
    plt.show()

# calcolo probabilità - guaussiana
print("\n")
print("\nValori t:")
[print(rf"t per $R_1$: {t_val[i]}") for i, t in enumerate(t_val)]

# definisco una funzione che vada a fare l'ODR dei dati dopo la frequenza di taglio
def FitRettaObliqua(x, y):
    modello = Model(lambda B, x: B[0] * x + B[1])  
    dati = RealData(x, y)
    odr = ODR(dati, modello, beta0=[0, np.mean(y)])
    output = odr.run()
    return output.beta, output.sd_beta  

# cerco parametri delle retta obliqua
Fit = [ FitRettaObliqua(xGrafico[i][22:29], yGrafico[i][22:29]) for i in range(len(Inv))]

for i in range(len(Fit)):
    print(f"m{i+1} =",  Fit[i][0][0], "+-", Fit[i][1][0])
    print(f"q{i+1} = ", Fit[i][0][1], "+-", Fit[i][1][1])

# stampo valori del coefficiente angolare della retta obliqua
print("\n")
[print(rf"Il coefficiente angolare $\rho$della retta obliqua per $R_1$ = {R1[i]} è pari a {fit[0][0]} $\pm$ {fit[1][0]} ") for i, fit in enumerate(Fit)]


# definisco la funzione che mi trovifrequenza di taglio

def FrequenzaTaglio(Alab, a, q):
    return 10**( ( 20 * np.log10(Alab) - q) / a )

# definisco la funzione per calcolare l'incertezza sulla frequenza di taglio

def incFrequenzaTaglio(A, a, q, incA, inca, incq):
    f = FrequenzaTaglio(A, a, q)
    
    # Derivate parziali
    rispettoA = f * (20 / (a * A * np.log(10)))
    rispettoa = -f * (20 * np.log10(A) - q) / (a**2)
    rispettoq = -f / (a * np.log(10))
    
    # Propagazione degli errori
    return np.sqrt((rispettoA * incA/2)**2 + (rispettoa * inca/2)**2 + (rispettoq * incq/2)**2)


# calcolo frequenze di taglio
FreqTaglio = [FrequenzaTaglio(A, Fit[i][0][0], Fit[i][0][1] ) for i, A in enumerate(Alab)] # / np.sqrt(2)

# calcolo incertezze nelle frequenze di taglio
incFreqTaglio = [incFrequenzaTaglio(A, Fit[i][0][0], Fit[i][0][1], incAlab[i], Fit[i][1][0], Fit[i][1][1] ) for i, A in enumerate(Alab)]

# stampa
print("\n")
[print(rf"La frequenza di taglio per $R_2$ = {R1[i]} è: ", Taglio, r"$\pm$" , incFreqTaglio[i] ) for i, Taglio in enumerate(FreqTaglio)]

# Stampo stampabilmente tutto lo stampabile
for i in range(len(Inv)):
    plt.errorbar(xGrafico[i],   yGrafico[i] , xerr = errXgrafico[i], yerr = errYgrafico[i], fmt = "o", markersize=2, elinewidth=1, capsize=1, label="Dati sperimentali", color=colori[i])

    #costante 
    plt.plot( np.linspace(xGrafico[i][0], np.log10(FreqTaglio[i] ) + 0.5, 500), np.full( 500, dB_pot(Alab[i], 1) ),  label="Fit retta costante", color="peachpuff", alpha=0.8)
    #obliquo
    x = np.linspace(np.log10(FreqTaglio[i] ) - 0.2, xGrafico[i][-1], 500)
    plt.plot( x, [Fit[i][0][0]*xi + Fit[i][0][1] for xi in x] , label="Fit retta obliqua", color="peachpuff", alpha=0.8)
    
    # stampo frequenza di taglio 
    plt.axvline(np.log10(FreqTaglio[i]), color = 'black', alpha=0.3, linestyle = 'dashed', label="Frequenza di taglio")

    #additivi
    plt.title(fr"Grafico Frequenza-Guadagno per $R_1 =$ {R1[i]} $\Omega$" )
    plt.xlabel(r"$log_{10}(\mathrm{f}) \ (decade)$" )
    plt.ylabel(r"$20 \ log_{10}\left(\frac{V_{OUT}}{V_{IN}}\right) (dB)$" )
    plt.legend()
    plt.savefig(f"/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInv/{R1[i]}graficoConTutto.png")
    plt.show()

# calcolo GBP
GBP = [fre * Alab[i] / np.sqrt(2) for i, fre in enumerate(FreqTaglio)]

# calcolo incertezza GBP
incGBP = [ np.sqrt( ( fre * incAlab[i] / np.sqrt(2) / 2 )**2 + ( incFreqTaglio[i] * Alab[i] / np.sqrt(2) / 2)**2 ) for i, fre in enumerate(FreqTaglio) ]

# Stampo 
print("\n")
[print( rf"GBP per $R_1$ = {R1[i]}$\Omega$:" , gbp , r"$\pm$" , incGBP[i], "Hz") for i, gbp in enumerate(GBP)]

# fit andamento iperbolico
def fit_iperbolico(dati_x, dati_y):
    def funzione_modello(beta, x):
        return beta[0] / (x - beta[1]) + beta[2]

    modello = Model(funzione_modello)
    dati = RealData(dati_x, dati_y)
    odr = ODR(dati, modello, beta0=[1000.0, 0, 1.0])
    output = odr.run()
    return output.beta, output.sd_beta

# fit di (A, freq)
FitIper = fit_iperbolico(Alab, FreqTaglio)

colori2 = ["royalblue", "deeppink", "limegreen", "chocolate"]

# Grafici sovrapposti
plt.title(fr"Grafici sovrapposti Frequenza-Guadagno con $R_2 =$ {R2} $\Omega$")
for i in range(len(Inv)):
    plt.errorbar(xGrafico[i],   yGrafico[i] , xerr = errXgrafico[i], yerr = errYgrafico[i], fmt = "o", markersize=2, elinewidth=0.5, capsize=1, label=rf"$R_1$ = {R1[i]} $\Omega$", color=colori2[i])
plt.xlabel(r"$log_{10}(\mathrm{f}) \ (decade)$" )
plt.ylabel(r"$20 \ log_{10}\left(\frac{V_{OUT}}{V_{IN}}\right) (dB)$" )
plt.legend()
plt.savefig(f"/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInv/{R2}graficiSovrapposti.png")
plt.show()

# grafico GPB a confronto
x = [0,1,2,3]

plt.title(rf"Scarti GBP con $R_2 =$ {R2} $\Omega$")
for i in range(len(GBP)):
    plt.errorbar(x[i], GBP[i], yerr=incGBP[i], fmt="o", color=colori[i], capsize=5)
plt.xticks(x, R1)
plt.xlabel(r"$R_1 (\Omega)$")
plt.ylabel("GBP")
plt.savefig(f"/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInv/{R2}graficiGBP.png")
plt.show()

# grafico A - freq

coeff_iper = FitIper[0]

xvaliper = np.linspace(0.1, max(Alab)*1.1, 500)
yvaliper = coeff_iper[0] / (xvaliper - coeff_iper[1]) + coeff_iper[2]  

plt.title(fr"Grafico di confronto A-Freq")
plt.xlabel(r"A" )
plt.ylabel(r"$log_{10}(\mathrm{f}) \ (decade)$")
for i in range(len(Alab)):
    plt.errorbar(Alab[i], FreqTaglio[i], xerr= incAlab[i], yerr=incFreqTaglio[i], color=colori[i], capsize=3)
plt.plot(xvaliper, yvaliper, color="black", linestyle="--", label="Fit iperbolico")
plt.legend()
plt.savefig(f"/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInv/{R2}graficiIper.png")
plt.show()

# coefficiente di correlazione
def coeffCorr(x, y):
    xmean = np.mean(x)
    ymean = np.mean(y)
    num = np.sum( (x - xmean) * (y - ymean) )
    den = np.sqrt( np.sum( (x-xmean)**2 ) * np.sum( (y-ymean)**2 ) )
    return num / den

# calcolo coefficienti di correlazione lineare
r = [coeffCorr(xGrafico[i][22:29], yGrafico[i][22:29]) for i in range(len(R1))]

# stampo coefficienti di correlazione
print("\n")
for i in range(len(R1)):
    print(rf"Coefficienti di correlazione per $R_1$ = {R1[i]}$\Omega$: {r[i]}")

# grafico log
Fretta = FitRettaObliqua(np.log10(Alab), np.log10(FreqTaglio))[0]

plt.title(fr"Grafico di confronto log10(A)-log10(Freq)")
plt.xlabel(r"A" )
plt.ylabel(r"$log_{10}(\mathrm{f}) \ (decade)$")
for i in range(len(Alab)):
    plt.errorbar(np.log10(Alab[i]), np.log10(FreqTaglio[i]), xerr= incAlab[i] / (np.log(10) * Alab[i]), yerr=incFreqTaglio[i] / (FreqTaglio[i] * np.log(10)), color=colori[i], capsize=3)
plt.plot(np.log10(xvaliper), Fretta[0] * np.log10(xvaliper) + Fretta[1], color="black", linestyle="--", label="Fit retta")
plt.legend()
plt.savefig(f"/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInv/{R2}graficiRettaLog.png")
plt.show()

#
Fretta2=FitRettaObliqua(Alab, FreqTaglio)[0]
plt.title(fr"Grafico di confronto A-Freq")
plt.xlabel(r"A" )
plt.ylabel(r"$\nu_c \ (Hz) $")
for i in range(len(Alab)):
    plt.errorbar(Alab[i], FreqTaglio[i], xerr= incAlab[i] , yerr=incFreqTaglio[i], color=colori[i], capsize=3)
plt.plot(xvaliper, Fretta2[0] * xvaliper + Fretta2[1], color="black", linestyle="--", label="Fit retta")
plt.legend()
plt.savefig(f"/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione1/GraficiInv/{R2}graficiRetta.png")
plt.show()

print("Coefficiente rappresentativo: ", Fretta2[0])