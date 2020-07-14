import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sea
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import  accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict



#CARICAMENTO DEL DATASET
dati_crm=pd.read_csv("CRM_data.csv",sep=";")
print(dati_crm.head())
print(dati_crm.describe())
print(dati_crm.dtypes)

print("\n ")
print("LUNGHEZZA DATASET")
print(dati_crm.shape)

#GIA QUI POSSIAMO NOTARE CHE ABBIAMO IL TIPO FLOAT, CHE DOVREBBE ESSERE INTERO
#CHIARAMENTE FAREMO IL PREPROCESSING SUCCESSIVAMENTE

###########################################################################################################
#1 LA COLONNA ID VIENE ELIMINATA. CONTIENE DATI TUTTI DIVERSI E QUINDI IN FASE DI ANALISI MI AGGIUNGE ENTROPIA
#I DATI MANCANTI VENGONO RIEMPITI CON IL VALORE ZERO.

#SOSTITUZIONE CON LA MEDIA
#from sklearn.impute import SimpleImputer
#imp = SimpleImputer(missing_values=np.nan, strategy=”mean”)

#Quindi vado a sostitire i valori Nan con la media.

#PER FIRST AMOUNTH SPENT E NUMBER OF PRODUCT:
# 0- 0 SIGNIFICA CHE NON HANNO COMPRATO NULLA QUINDI LA RIGA VIENE ELIMINATA (SI PUO LAVORARE PER IL FITRAGGIO
#SU EXCEL

#1-0 E 0-1 SOSTITUISCO LA CELLA CON VALORE ZERO CON LA MEDIA

#PER LE FASCIE DI ETA CREO UNA COLONNA DOVE FACCIO LA SOMMA CHE DOVRA' DARMI UNO, ALTRIMENTI ELIMINO LA RIGA


#########################################################################################################
#QUI ANDIAMO A CARICARE IL DATA SET PREPROCESSATO
dati_crm_preprocessed=pd.read_csv("CRM_data2.csv",sep=";")
####################################
#STATISTICHE ELABORATE.
####################################
# 1) NUMERO DI CLIENTI CON IL PAGAMENTO RATEALE
installment=dati_crm_preprocessed[dati_crm_preprocessed['installment']==1]

print("\n Analisi Spesa a rate ")
acquisto_rate=installment['first_amount_spent']
print(acquisto_rate.describe())

#3) PERSONE CHE NON HANNO FATTO SPESA A RATE
print("\n Analisi Spesa NON a rate ")
installment_no_rate=dati_crm_preprocessed[dati_crm_preprocessed['installment']==0]
acquisto_rate0=installment_no_rate['first_amount_spent']
print(acquisto_rate0.describe())

#DIFFERISCONO LA SPESA MINIMA E IL 50%

#ANALISI DEL NUMERO DI PRODOTTO
num_prodotti=dati_crm_preprocessed['first_amount_spent']
#istogramma=num_prodotti.hist(bins=50)


#SETTAGGIO DEGLI ASSI
#istogramma.set_title("distribuzione spesa-numero prodotti primo aquisto")
#istogramma.set_xlabel("valore della spesa")
#istogramma.set_ylabel("numero prodotti")

# DALLE API DI MATLPLOTLIB.PYPLOT, E' POSSIBILE ELININARE LA NOTAZIONE SCIENTIFICA
#plt.ticklabel_format(style='plain')
#plt.show()

#DA QUI, TROVATA LA SPESA EFFETTUATA DAL CLIENTE NEL PRIMO ACQUISTO
#ANDIAMO A CALCOLARE LA SOMMA (VALORE COMPLESSIVO PER I DIVERSI VALORI DI NUMBER)
number_product=dati_crm_preprocessed[['number_of_product','first_amount_spent']].groupby('number_of_product')
somma=number_product.sum().sort_values('first_amount_spent')
somma.plot.barh()
#plt.show()

#FACCIAMO LA STESSA COSA PER AG
eta_51_89=dati_crm_preprocessed[['age51_89','first_amount_spent']].groupby('age51_89')
somma1=eta_51_89.sum().sort_values('first_amount_spent')
#somma1.plot.barh()
#plt.show()

eta_36_50=dati_crm_preprocessed[['age36_50','first_amount_spent']].groupby('age36_50')
somma2=eta_36_50.sum().sort_values('first_amount_spent')
#somma2.plot.barh()
#plt.show()

eta_15_35=dati_crm_preprocessed[['age15_35','first_amount_spent']].groupby('age15_35')
somma3=eta_15_35.sum().sort_values('first_amount_spent')
#somma3.plot.barh()
#plt.show()

#AGENZIA
agenzia_nord=dati_crm_preprocessed[['north','first_amount_spent']].groupby('north')
somma4=agenzia_nord.sum().sort_values('first_amount_spent')
#somma4.plot.barh()
#plt.show()

agenzia_centro=dati_crm_preprocessed[['center','first_amount_spent']].groupby('center')
somma5=agenzia_centro.sum().sort_values('first_amount_spent')
#somma5.plot.barh()
#plt.show()

agenzia_sud=dati_crm_preprocessed[['south_and_islands','first_amount_spent']].groupby('south_and_islands')
somma6=agenzia_sud.sum().sort_values('first_amount_spent')
#somma6.plot.barh()
#plt.show()



#NON ESSENDO PRESENTI DATI CATEGORIALI , POSSIAMO PROSEGUIRE LA NOSTRA ANALISI
#MATRICE DI CORRELAZIONE PER CAPIRE COME SI HA CORRELAZIONE CON LA VARIABILE TARGET
#USIAMO LA FUNZIONE CORR() OFFERTA DA PANDAS

#PER UNA VISIONE GRAFICA IMPORTIAMO SEABORN

matrice_correlazione=dati_crm_preprocessed.corr()
#CREAZIONE DI UNA FIGURA
#plt.figure(figsize=(10,20))

#MATRICE DI CONFUSIONE (LIBRERIA SEABORN)

#PARAMETRI PASSATI:
#1)  LA MATRIC
#2) VMAX(SFUMATURE),
#3)SQUARE(OGNI CELLA HA FORMA QUADRRATA)
#4) ANNOT SCRIVE IL VALORE IN OGNI CELLA
#5) FMT con 2f sono il numero di decimali
#6) CMAP E' IL COLORE (esempio: plasma, magma,cividis)
#7) CBAR-kws ARGOMENTI PAROLE CHIAVE
#8) ROBUST SE E' TRUE E NON CI STA VMAX E VMIN VIENE CALCOLATO CON QUANTII ROBUSTI



grafica= sea.heatmap(matrice_correlazione,vmax=2,square=True,annot=True,fmt='.2f',
                     cmap='magma',cbar_kws={"shrink": .5},robust=True)
plt.title('MATRICE DI CORRELAZIONE',fontsize=20)
#plt.show()


#CORRELAZIONE TRA VARIABILE TARGET E ALTRE VARIAILI
print("\n \n ----------------------------------------")
print("CORRELAZIONE TRA Y E ALTRE VARIABILI ")
x=dati_crm_preprocessed.corr()
print(x["Y"].sort_values(ascending=False))

#plt.show()

print("--------------------------------------------------")



data=pd.read_csv("CRM_data2.csv",sep=";")

###################################################################################
#CON .VALUES RITORNIAMO LA VERSIONE NUMPY DEL DATAFRAME FORNITO
#NUMPY E' LA LIBRERIA CHE SERVE PER LA GESTIONE DI ARRAY E MATRICI

array=data.values
###################################################################################
x=array[:]  #E' UNA COPIA DELL ARRAY
print("stampa degli array \n ",x)

###################################################################################
#COLONNE  DELLA VARIABILE TARGHET
###################################################################################
y= array[:,3]  #sono le coordinate della variabile targhet (quindi prendo
#le colonne)
print("stampa di Y ")
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
nomi_classificatori = ["Nearest Neigbours","Decision Tree","Random Forest","Naive Bayles", "Logistic Regression",
         "Gradient Boosting","Gaussian Process"]
###################################################################################################
#CREO UN ARRAY CON I CLASSIFICATORI UTILIZZATI. CHIARAMENTE VANNO IMPORTATE LE LIBRERIE APPOSITE
classifiers = [KNeighborsClassifier(3),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5,n_estimators=10,max_features=1),
               GaussianNB(),
               LogisticRegression(solver="lbfgs"),
               GradientBoostingClassifier(n_estimators=1000),
               GaussianProcessClassifier()]

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

#CROSS VALIDATION
#DIVIDIAMO I DATI IN SOTTOINSIEMI K CHE SONO CHIAMATI FOLD. SE IMMAGINIAMO
#DI DIVIDERE IN 10 FOLD IL MODELLO SCELTO VIENE ADDESTRATO  VALUTATO 10 VOLTE
#USANDO UN FOLD DIVERSO OGNI VOLTA
dt=DecisionTreeClassifier(max_depth=5)
score=cross_val_score(dt,x_train,y_train,cv=10,scoring="accuracy")
print("scores",score)
print("Media",score.mean())
print("deviazione standard",score.std())

#MATRICE DI CONFUSIONE
errori_classificatore= cross_val_predict(DecisionTreeClassifier(),x_train,y_train,cv=3)
matrice=confusion_matrix(y_train,errori_classificatore)
print("\n \n \n")
print("________________________________________________________")

print(matrice)

#sulla diagonale ci sono le classifixazioni corrette
print("________________________________________________________")

print("PRECISION",precision_score(y_train,errori_classificatore))
print("________________________________________________________")
print("RECALL",recall_score(y_train,errori_classificatore))
print("________________________________________________________")