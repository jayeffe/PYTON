import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sea
import  sklearn as sk
from sklearn import  preprocessing

from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import  accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict




#CARICAMENTO DEL DATASET
dati_nursery=pd.read_csv("nursery data.csv",sep=";")
print(dati_nursery.head())
print(dati_nursery.describe())
print(dati_nursery.dtypes)

print("\n ")
print("LUNGHEZZA DATASET")
print(dati_nursery.shape)


#I DATI PRESENTI SONO TUTTI ETEROGENEI DI TIPO OBJECT.
#NON SONO SCAMBIATI PER TESTUALI MA SONO CATEGORIALI
#CHILDREN INVECE HA DATI SIA TESTUALI CHE NUMERICI

#-------------------------------------------------------------------
#PREPROCESSING
#-------------------------------------------------------------------
#NON CI SONO DATI INCOMPLETI, CELLE VUOTE, O INCONSISTENTI
#DAL DATASET NON TOGLIAMO NULLA, NON VI SONO ATTRIBUTI CHE INTRODUCONO
#ENTROPIA, AD ESEMPIO (CODICI ID TUTTI DIVERSI)

#-------------------------------------------------------------------
#DISCRETIZZAZIONE
#-------------------------------------------------------------------

newdata=dict() #collezione  coppie  chiave-valore, ogni coppia
#chiave valore, mappa la chiave al valore associato


#con un ciclo for controlliamo le chiavi:
print("CHIAVI DEL DICT() ")
for chiavi in dati_nursery.keys():
  print(chiavi)


print("\n ")

#CREATO IL DIZIONARIO TRAMITE UN CICLO FOR CONTROLLO PER TUTTE
#LE COLONNE IL TIPO OGGETTO
#PER TRASFORMARE UTILIZZO IL LABEL ENCODER SUL TIPO OBJECT
#TRAMITE IF.


for chiavi in dati_nursery.keys():
    if(dati_nursery[chiavi].dtypes=='object'):
        le=preprocessing.LabelEncoder()
        le.fit(dati_nursery[chiavi])
        newdata[chiavi]=le.transform(dati_nursery[chiavi])
    else:
        newdata[chiavi]=dati_nursery[chiavi]
#CONFERMIAMO A PANDAS IL NUOVO DATAFRAME
new= pd.DataFrame.from_dict(newdata,orient='columns',dtype=None)
print(new.head())




print("-------------------------------------------------------------------")
#POSSIAMO PROSEGUIRE LA NOSTRA ANALISI
#MATRICE DI CORRELAZIONE PER CAPIRE COME SI HA CORRELAZIONE CON LA VARIABILE TARGET
#USIAMO LA FUNZIONE CORR() OFFERTA DA PANDAS

#PER UNA VISIONE GRAFICA IMPORTIAMO SEABORN

matrice_correlazione=new.corr()
#CREAZIONE DI UNA FIGURA
#POSSIAMO PROSEGUIRE LA NOSTRA ANALISI
#MATRICE DI CORRELAZIONE PER CAPIRE COME SI HA CORRELAZIONE CON LA VARIABILE TARGET
#USIAMO LA FUNZIONE CORR() OFFERTA DA PANDAS

#PER UNA VISIONE GRAFICA IMPORTIAMO SEABORN

matrice_correlazione=new.corr()
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


#CORRELAZIONE TRA VARIABILE TARGET E ALTRE VARIAILI
print("\n \n ----------------------------------------")
print("CORRELAZIONE TRA Y E ALTRE VARIABILI ")
x=new.corr()
print(x["class"].sort_values(ascending=False))

plt.show()

print("--------------------------------------------------")
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
y= array[:,8]  #sono le coordinate della variabile targhet (quindi prendo
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
nomi_classificatori = ["Nearest Neigbours","Decision Tree","Random Forest","Naive Bayles",
         "Gradient Boosting"]
###################################################################################################
#CREO UN ARRAY CON I CLASSIFICATORI UTILIZZATI. CHIARAMENTE VANNO IMPORTATE LE LIBRERIE APPOSITE
classifiers = [
               KNeighborsClassifier(3),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5,n_estimators=10,max_features=1),
               GaussianNB(),
               GradientBoostingClassifier(n_estimators=1000)
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


######################################################################################################
#CROSS VALIDATION
#DIVIDIAMO I DATI IN SOTTOINSIEMI K CHE SONO CHIAMATI FOLD. SE IMMAGINIAMO
#DI DIVIDERE IN 10 FOLD IL MODELLO SCELTO VIENE ADDESTRATO  VALUTATO 10 VOLTE
#USANDO UN FOLD DIVERSO OGNI VOLTA
print ("\n **********\n Risultati K neibourds \n *********\n")

dt=KNeighborsClassifier()
score=cross_val_score(dt,x_train,y_train,cv=10,scoring="accuracy")
print("Punteggio ",score)
print("Media",score.mean())
print("deviazione standard",score.std())


print ("\n **********\n Risultati cross validation \n *********\n")
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, x_train, y_train, cv=10, scoring = "accuracy")
print("Punteggio: ", scores)
print("Media: ", scores.mean())
print("Deviazione standard: ", scores.std())


#print ("\n **********\n Risultati Gradient Booster \n *********\n")
#from sklearn.model_selection import cross_val_score
#rf =GradientBoostingClassifier(n_estimators=1000)
#scores = cross_val_score(rf, x_train, y_train, cv=10, scoring = "accuracy")
#print("Punteggio: ", scores)
#print("Media: ", scores.mean())
##print("Deviazione standard: ", scores.std())


# MATRICE DI CONFUSIONE
errori_classificatore= cross_val_predict(GradientBoostingClassifier(n_estimators=100),x_train,y_train,cv=3)
matrice=confusion_matrix(y_train,errori_classificatore)
print("\n \n \n")
print("________________________________________________________")

#0 = not_recom
#1 = priority
#2 = recommend
#3 = spec_prior
#4 = very_recom

print(matrice)
print("________________________________________________________")

print("PRECISION",precision_score(y_train,errori_classificatore,average=None))
print("________________________________________________________")
print("RECALL",recall_score(y_train,errori_classificatore,average=None))
print("________________________________________________________")


# MATRICE DI CONFUSIONE
errori_classificatore= cross_val_predict(KNeighborsClassifier(),x_train,y_train,cv=3)
matrice=confusion_matrix(y_train,errori_classificatore)
print("\n \n \n")
print("________________________________________________________")

#0 = not_recom
#1 = priority
#2 = recommend
#3 = spec_prior
#4 = very_recom

print(matrice)
print("________________________________________________________")

print("PRECISION",precision_score(y_train,errori_classificatore,average=None))
print("________________________________________________________")
print("RECALL",recall_score(y_train,errori_classificatore,average=None))
print("________________________________________________________")


