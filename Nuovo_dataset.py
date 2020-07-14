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
dati_nursery=pd.read_csv("nursery_copy.csv",sep=";")

newdata=dict() #collezione  coppie  chiave-valore, ogni coppia



for chiavi in dati_nursery.keys():
    if(dati_nursery[chiavi].dtypes=='object'):
        le=preprocessing.LabelEncoder()
        le.fit(dati_nursery[chiavi])
        newdata[chiavi]=le.transform(dati_nursery[chiavi])
    else:
        newdata[key]=dati_nursery[key]
#CONFERMIAMO A PANDAS IL NUOVO DATAFRAME
new= pd.DataFrame.from_dict(newdata,orient='columns',dtype=None)
print(new.head())

print("-------------------------------------------------------------------")

matrice_correlazione=new.corr()


rafica= sea.heatmap(matrice_correlazione,vmax=1,square=True,annot=True,fmt='.2f',
                     cmap='magma',cbar_kws={"shrink": .5},robust=True)
plt.title('MATRICE DI CORRELAZIONE',fontsize=20)
plt.show()


#CORRELAZIONE TRA VARIABILE TARGET E ALTRE VARIAILI
print("\n \n ----------------------------------------")
print("CORRELAZIONE TRA Y E ALTRE VARIABILI ")
x=new.corr()
print(x["health"].sort_values(ascending=False))

plt.show()

print("--------------------------------------------------")
###################################################################################
#CON .VALUES RITORNIAMO LA VERSIONE NUMPY DEL DATAFRAME FORNITO
#NUMPY E' LA LIBRERIA CHE SERVE PER LA GESTIONE DI ARRAY E MATRICI

array=new.values
###################################################################################
x=array[:]  #E' UNA COPIA DELL ARRAY

###################################################################################
#COLONNE  DELLA VARIABILE TARGHET
###################################################################################
y= array[:,8]  #sono le coordinate della variabile targhet (quindi prendo
#le colonne)
###################################################################################
#DALLA LIBRERIA SKLEARN MODEL SELECTION ANDIAMO A DIVIDERE IN TRAIN E TEST

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#quindi andiamo a passare il dataset delle istanze di x
#target: il dataset delle etichette
#test size, 0.3 perchè il 30 percento è per il test, ma potevo usare anche il 70,20
#random state e' l'indicatore di randomizzaione

#OUTPUT
#  INSIEME DI ATDDESTRAMENTO
#  INSIEME DI TEST
# Y TRAIN SONO LE ETICHETTE DELL'INSIEME DI ADDESTRAMENTO
#Y RTET SONO LE ETICHETTE DEL INSIEME DI TEST

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
dt=KNeighborsClassifier()
score=cross_val_score(dt,x_train,y_train,cv=10,scoring="accuracy")
print("scores",score)
print("Media",score.mean())
print("deviazione standard",score.std())

# MATRICE DI CONFUSIONE
errori_classificatore= cross_val_predict(KNeighborsClassifier(),x_train,y_train,cv=3)
matrice=confusion_matrix(y_train,errori_classificatore)
print("\n \n \n")
print("________________________________________________________")

#0 = not_recom
#1 = priority
#2 = recommend
#3 = spec_prior


print(matrice)
print("________________________________________________________")

print("PRECISION",precision_score(y_train,errori_classificatore,average=None))
print("________________________________________________________")
print("RECALL",recall_score(y_train,errori_classificatore,average=None))
print("________________________________________________________")

