
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk


from sklearn import preprocessing

from sklearn.preprocessing import  LabelEncoder
print("-----------------------------------")

letto=pd.read_csv("posti_letto.csv",sep=";")
print("PRIME 5 RIGHE ")
print(letto.head())
print("-----------------------------------")
print(letto.describe())
print("-----------------------------------")

print(letto.dtypes)
######################################################################################################
#STATISTICHE ELABORATE.

# 1) OSPEDALI CENSITI NEL 2011
anno2011=letto[letto['Anno']==2011]
print("----------------------------------------")
print("\n Analisi ospedali censiti nel 2011  ")
ospedali=anno2011['Denominazione struttura']
print(ospedali)
print("-----------------------------------")

print("STATISTICA SUI POSTI LETTO ANNO 2011")
letti=anno2011['Totale posti letto']
print(letti.describe())

istogramma = letti.hist(bins=30)

istogramma.set_title('Relazione tra numero di letti e numero di ospedali')
istogramma.set_xlabel('Numero letti')
istogramma.set_ylabel('Numero ospedali')
plt.show()
###########################################################################################################
#EMERGONO TANTI OSPEDALI CON POCHI POSTI LETTO

##########################################################################################################
anno_2011=letto[letto['Anno']==2011]
year_2011=anno2011.sort_values('Totale posti letto',ascending=False)
#########################################################################################################
#stampa i soli attributi 'Denominzione struttura' e 'Totale posti letto' del dataset anno
#######################################################################################################
print("--------------------------------------------------------------------------------------")

print(letto[['Denominazione struttura', 'Totale posti letto']])

print("--------------------------------------------------------------------------------------")
######################################################################################################
#ANALISI PIU ACCURATA , RAGGRUPANDO IL CONTEGGIO PER REGIONI E POI SOMMANDO
#####################################################################################################
letti_per_regione=letto[['Descrizione Regione','Totale posti letto']].groupby('Descrizione Regione')
somma_ordina=letti_per_regione.sum().sort_values('Totale posti letto')

somma_ordina.plot.barh()
plt.show()