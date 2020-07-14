# Il dataset Bank Data contiene informazioni bancarie e personali di alcuni clienti.
#Si vogliono definire modelli di  targeting per classificare i clienti e vedere la distribuzine del reddito, in base all'acquisto del PEP (Personal
# Equity Plan)




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sea

import sklearn as sk

from sklearn import preprocessing

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import  accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict


#CARICAMENTO DEL DATASET
dati_bank=pd.read_csv("bank-data.csv",sep=";")
print("---------------------------------------------")
print(dati_bank.head())
print("---------------------------------------------")

print(dati_bank.describe())
print("---------------------------------------------")

print(dati_bank.dtypes)
print("---------------------------------------------")

print("\n ")
print("LUNGHEZZA DATASET")
print(dati_bank.shape)

#PREPROCESSING
#1) VERIFICA ID CODE DUPLICATI, ANCHE SE POI LI ELIMINIAMO PERCHE' AGGIUNGONO ENTROPIA
#2) VERIFICA ATTRIBUTO AGE DI MASSIMO E MINIMO. I DATI POSSONO ANDARE BENE. SONO ETA' CONFORMI
#3) GLI ALTRI ATTRIBUTI SONO CORRETTI
#4) I DATI DI INCOME SONO INACCURATI. CI SONO 25 DATI CHE NON VENGONO SCRITTI BENE INFATTI COMPARE
#DUE VOLTE LA VIRGOLA, PERTANTO SONO SOSTITUITI DAL VALORE MEDIO 28348,4

dati_bank2=pd.read_csv("bank-data2.csv",sep=";")
print("\n ---------------------------------------------")
print("DATI PREPROCESSATI ")
print("---------------------------------------------")

print(dati_bank2.head())
print("---------------------------------------------")

print(dati_bank2.describe())
print("---------------------------------------------")

print(dati_bank2.dtypes)
print("---------------------------------------------")

#notiamo che compare il valore medio scelto da noi

# Normalizzazione delle fasce di età:
# ● 0 = intervallo [18, 40]
# ● 1 = intervallo [40, 60]
# ● 2 = intervallo [60, 80]
# ● 3 = intervallo [80, 100]

#prendo le istanze di tipo 'age' dal dataset data
age= dati_bank2['age']
#metto i valori in una lista ( ARRAY 1 DIMENSIONE ]
title= pd.Series(['age'])
new_age= pd.cut(age,[18,40,60,80,100],labels=False,right=False)
print ("stampa delle nuove fasce d'età per ogni persona: \n *********\n")
print(new_age)

#apro il dataset age.csv, inserisco la nuova colonna new_age e lo chiudo
f = open("age.csv", "a") #'a' crea un nuovo file se non esiste
new_age.to_csv('age.csv', index = False )
f.close()

data_con_newage = pd.read_csv("age.csv", sep=';')

ist1 = data_con_newage['age'].hist(bins=50)
ist1.set_title("distribuzione fascie d'età")
ist1.set_xlabel("intervalli ")
ist1.set_ylabel("Numero persone ")

plt.show()

#STATISTICHE TRAMITE ISTOGRAMMI E DISTRIBUZIONI
#REDDITO IN RELAZIONE AL SESSO
dati_donna= dati_bank2[dati_bank2['sex']=='FEMALE']
data_income= dati_donna['income']
print("-------------------------------------------------")
print("ANALISI REDDITO SESSO=FEMALE ")
print("-------------------------------------------------")
print(data_income.describe())

dati_uomo= dati_bank2[dati_bank2['sex']=='MALE']
data_income_man= dati_uomo['income']
print("-------------------------------------------------")
print("ANALISI REDDITO SESSO=MALE ")
print("-------------------------------------------------")
print(data_income_man.describe())

#ANALISI REDDITO- SESSO FEMALE -REGIONE
dati_regione=dati_donna[['region','income']].groupby('region')
somma=dati_regione.sum().sort_values('income')
somma.plot.barh()
plt.xlabel("ANALISI REDDITO IN BASE AL SESSO FEMMINILE PER REGIONE")
plt.show()
#ANALISI REDDITO- SESSO MALE -REGIONE
dati_regione=dati_uomo[['region','income']].groupby('region')
somma=dati_regione.sum().sort_values('income')
somma.plot.barh()
plt.xlabel("ANALISI REDDITO IN BASE AL SESSO MASCHILE PER REGIONE ")
plt.show()

# I CLIENTI CON REDDITO PIU' ALTO SONO NEL CENTRO CITTA'
#REDDITO PIU BASSO SONO IN PERIFERIA

# ANALISI SIMILI possono essere fatte per  valutare ad esempio
# distribuzione del reddito in base al matrimonio e al numero dei ﬁgli.

children=dati_bank2[dati_bank2['children']==0]
dati_children=dati_bank2[['region','income','married']].groupby('region')
somma2=dati_children.sum().sort_values('income')
somma2.plot.barh()
plt.xlabel("ANALISI REDDITO IN BASE AL MATRIMONIO E NUMERO DEI FIGLI ")
plt.show()

print("\n")
print("--------------------------------------------------")
print("STATISTICHE SUL REDDITO CLIENTI SENZA FIGLI")
print(somma2.describe)



#ANDIAMO A CONOSCERE IL NUMERO DI CLIENTI PER OGNI REGIONE
inner_city=dati_bank2[dati_bank2['region']=='INNER_CITY']
numero_inner_city=len(inner_city)
print("\n clienti che abitano in inner city")
print(numero_inner_city)

town= dati_bank2[dati_bank2['region']=='TOWN']
numero_town=len(town)
print("\n clienti che abitano in town")
print(numero_town)

rural= dati_bank2[dati_bank2['region']=='RURAL']
numero_rural= len(rural)
print("\n persone che abitano in rural")
print(numero_rural)

suburban=dati_bank2[dati_bank2['region']=='SUBURBAN']
num_suburban=len(suburban)
print("\n persone che abitano in suburban ")
print(num_suburban)

print("-------------------------------------------------------------")
#-------------------------------------------------------------------
#DISCRETIZZAZIONE
#-------------------------------------------------------------------

newdata=dict() #collezione  coppie  chiave-valore, ogni coppia
#chiave valore, mappa la chiave al valore associato


#con un ciclo for controlliamo le chiavi:
print("CHIAVI DEL DICT() ")
for chiavi in dati_bank2.keys():
  print(chiavi)


print("\n ")

#CREATO IL DIZIONARIO TRAMITE UN CICLO FOR CONTROLLO PER TUTTE
#LE COLONNE IL TIPO OGGETTO
#PER TRASFORMARE UTILIZZO IL LABEL ENCODER SUL TIPO OBJECT
#TRAMITE IF.



for chiavi in dati_bank2.keys():
    if(dati_bank2[chiavi].dtypes=='object'):
        le=preprocessing.LabelEncoder()
        le.fit(dati_bank2[chiavi])
        newdata[chiavi]=le.transform(dati_bank2[chiavi])
    else:
        newdata[chiavi]=dati_bank2[chiavi]
#CONFERMIAMO A PANDAS IL NUOVO DATAFRAME
new= pd.DataFrame.from_dict(newdata,orient='columns',dtype=None)
print(new.head())

print("-------------------------------------------------------------------")
#MATRICE DI CORRELAZIONE PER CAPIRE COME SI HA CORRELAZIONE CON LA VARIABILE TARGET
#USIAMO LA FUNZIONE CORR() OFFERTA DA PANDAS

#PER UNA VISIONE GRAFICA IMPORTIAMO SEABORN
matrice_correlazione=new.corr()
#CREAZIONE DI UNA FIGURA
#POSSIAMO PROSEGUIRE LA NOSTRA ANALISI
#MATRICE DI CORRELAZIONE PER CAPIRE COME SI HA CORRELAZIONE CON LA VARIABILE TARGET
#USIAMO LA FUNZIONE CORR() OFFERTA DA PANDAS

#PER UNA VISIONE GRAFICA IMPORTIAMO SEABORN

#CREAZIONE DI UNA FIGURA
plt.figure(figsize=(10,20))

#MATRICE DI CONFUSIONE (LIBRERIA SEABORN)

#PARAMETRI PASSATI:
#1)  LA MATRIC
#2) VMAX(PER ANCORARE LA MAPPA DEI COLORI),
#3)SQUARE(OGNI CELLA HA FORMA QUADRRATA)
#4) ANNOT SCRIVE IL VALORE IN OGNI CELLA
#5) FMT con 2f sono il numero di decimali
#6) CMAP E' IL COLORE (esempio: plasma, magma,cividis)
#7) CBAR-kws ARGOMENTI PAROLE CHIAVE
#8) ROBUST SE E' TRUE E NON CI STA VMAX E VMIN VIENE CALCOLATO CON QUANTII ROBUSTI



grafica= sea.heatmap(matrice_correlazione,vmax=1,square=True,annot=True,fmt='.2f',
                     cmap='magma',cbar_kws={"shrink": .5},robust=True)
plt.title('MATRICE DI CORRELAZIONE',fontsize=20)
plt.show()

# La matrice di correlazione mostra che l'unica variabile correlata con la target
# è 'income'(reddito del cliente), anche se la correlazione è comunque bassa (0.21)
# Si può notare però che le due variabili in maggior correlazione sono income ed age (0.17).


#CORRELAZIONE TRA VARIABILE TARGET E ALTRE VARIAILI
print("CORRELAZIONE TRA Y E ALTRE VARIABILI ")
x=new.corr()
print(x["pep"].sort_values(ascending=False))

plt.show()

###################################################################################
#CON .VALUES RITORNIAMO LA VERSIONE NUMPY DEL DATAFRAME FORNITO
#NUMPY E' LA LIBRERIA CHE SERVE PER LA GESTIONE DI ARRAY E MATRICI

array=new.values
###################################################################################
x=array[:]  #E' UNA COPIA DELL ARRAY
print("stampa degli array \n ",x)

###################################################################################
#COLONNE  DELLA VARIABILE TARGHET
###################################################################################
y= array[:,10]  #sono le coordinate della variabile targhet (quindi prendo
#le colonne)
print("stampa ISTANZE DI CLASS (TARGHET) ")
print(y)
###################################################################################
#DALLA LIBRERIA SKLEARN MODEL SELECTION ANDIAMO A DIVIDERE IN TRAIN E TEST

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#quindi andiamo a passare il dataset delle istanze di x
#target: il dataset delle etichette
#test size, 0.3 perchè il 30 percento è per il test, ma potevo usare anche il 70,20
#random state e' l'indicatore di randomizzaione

#OUTPUT
#  INSIEME DI ATDDESTRAMENTO
#  INSIEME DI TEST
# Y TRAIN SONO LE ETICHETTE DELL'INSIEME DI ADDESTRAMENTO
#Y RTET SONO LE ETICHETTE DEL INSIEME DI TEST
###################################################################################
#STAMPIAMO I VALORI TRAMITE LO SHAPE
print("\n \n \n ")
print("------------------------------- ")

print("SHAPE OF X : ",x.shape)
print("SHAPE OF Y : ",y.shape)
print("------------------------------- ")



print("SHAPE OF X train: ",x_train.shape)
print("SHAPE OF Y train: ",y_train.shape)
print("------------------------------- ")

print("SHAPE OF X test: ",x_test.shape)
print("SHAPE OF Y test: ",y_test.shape)
print("------------------------------- ")
###################################################################################################
#CREO UN ARRAY CON I NOMI DEI CLASSIFICATORI
nomi_classificatori = ["Nearest Neigbours","Decision Tree","Random Forest","Naive Bayles","Logistic Regession"
         "Gradient Boosting", "Gaussian Process"]
###################################################################################################

#CREO UN ARRAY CON I CLASSIFICATORI UTILIZZATI. CHIARAMENTE VANNO IMPORTATE LE LIBRERIE APPOSITE
classifiers = [
               KNeighborsClassifier(3),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5,n_estimators=10,max_features=1),
               GaussianNB(),
               LogisticRegression(solver="lbfgs"),
               GradientBoostingClassifier(n_estimators=1000),
               GaussianProcessClassifier()
               ]
########################################################################################################
#IN QUESTO MODO QUA ANZICHE' SCORRERE IL SINGOLO ARRAY LI SCORRO A COPPIA
for nomi_classificatori, x in zip(nomi_classificatori,classifiers):
    print(nomi_classificatori)
#ADDESTRAMENTO DEGLI ALGORITMI
    x.fit(x_train,y_train)
#PREVISIONE DEGLI ALGORITMI SUI NOSTRI DATI DI TEST
    y_predict=x.predict(x_test)
#STAMPIAMO L'ACCURACY SCORE PER L'ACCURATEZZA DEL CLASSIFICATORE
    print(accuracy_score(y_test,y_predict))
    print("ACCURATEZZA %0.2f\n"  % accuracy_score(y_test,y_predict))

print("--------------------------------------------------------------------")


####################################################################################################
######################################################################################################
#CROSS VALIDATION
#DIVIDIAMO I DATI IN SOTTOINSIEMI K CHE SONO CHIAMATI FOLD. SE IMMAGINIAMO
#DI DIVIDERE IN 10 FOLD IL MODELLO SCELTO VIENE ADDESTRATO  VALUTATO 10 VOLTE
#USANDO UN FOLD DIVERSO OGNI VOLTA
print ("\n Risultati cross validation alberi ")

dt=DecisionTreeClassifier(max_depth=5)
score=cross_val_score(dt,x_train,y_train,cv=10,scoring="accuracy")
print("Punteggio ",score)
print("Media",score.mean())
print("deviazione standard",score.std())

print("--------------------------------------------------------------------")
print ("\n Risultati cross validation ")
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, x_train, y_train, cv=10, scoring = "accuracy")
print("Punteggio: ", scores)
print("Media: ", scores.mean())
print("Deviazione standard: ", scores.std())

print("--------------------------------------------------------------------")
errori_classificatore= cross_val_predict(DecisionTreeClassifier(),x_train,y_train,cv=3)
matrice=confusion_matrix(y_train,errori_classificatore)
print("\n ")
print("________________________________________________________")
print("MATRICE DI CONFUSIONE ")
print(matrice)
print("________________________________________________________")

print("PRECISION",precision_score(y_train,errori_classificatore,average=None))
print("________________________________________________________")
print("RECALL",recall_score(y_train,errori_classificatore,average=None))
print("________________________________________________________")

