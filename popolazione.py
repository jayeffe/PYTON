#Per fare una statica completa dei posti letto per regione
#ci serve anche la popolazione per regione
#che non è presente nel database originario.
#Quindi creiamo un nuovo database con: codice regione e popolazione.
#Dobbiamo fare la  combinazione di due database.

import pandas as pd
import matplotlib.pyplot as plt

letto= pd.read_csv("posti_letto.csv",sep=";")
anno=letto[letto['Anno']==2011]

#qua faccio una selezione multipla: notiamo che uso una coppia di parentesi quadre

letti=anno[['Codice Regione','Descrizione Regione','Totale posti letto']].groupby(['Codice Regione'])


#aggregazione dei dati (il numero di posti letto viene sommato)
letti = letti.aggregate( { 'Descrizione Regione':'first', 'Totale posti letto':'sum' } )

print ("\n **********\n Dati aggregati: \n *********\n")
print(letti)

# Carica il secondo csv 'popolazione.csv'
dati2 = pd.read_csv( 'popolazione.csv', sep=';' )


# Raggruppo i cittadin per Codice regione e sommo tutti i loro cvalori.
# Il risultato e'¨ la lista delle regioni con associato il numero di abitanti
popolazione = dati2[ ['Codice Regione', 'Popolazione' ] ].groupby( 'Codice Regione' ).sum()

print(popolazione)

# join tra i dataframe popolazione e letti.
# Serve almeno una variabile con lo stesso nome (nel nostro caso  Codice Regione).

#quindi avrro codice regione e popolazione che fanno parte di un dataframe
#descrizione regione e totale posti letto che fanno parte dell'altro dataframe

lettiEPopolazione = popolazione.join( letti )
print("letti e popolazione")
print(lettiEPopolazione)

# Aggiungiamo una colonna: 'Letti per Cittadino' che e' data da 'Totale posti letto'/'Popolazione'
lettiEPopolazione['Letti per Cittadino'] = lettiEPopolazione['Totale posti letto'] / lettiEPopolazione['Popolazione']

# Ordinamento del nuovo dataframe
ordinato = lettiEPopolazione.sort_values( 'Letti per Cittadino' )
print ("\n **********\n Dataframe ottenuto dal join e ordinato: \n *********\n")
print(ordinato)

print ("\n **********\n Statistiche descrittive dell' istanza Letti per Cittadino dopo il join: \n *********\n")
print (ordinato['Letti per Cittadino'].describe())




# Grafico, specifico tipo e cosa visuallizare sugli assi
ordinato.plot( kind='barh', x='Descrizione Regione', y='Letti per Cittadino' )
plt.show()


# la Lombardia, che sembrava possedere un numero spropositato di posti letto
#non è più neanche al primo posto.
#Le differenze tra le varie regioni sono attenuate
#Il molise da ultimo per posti letto diventa il primo