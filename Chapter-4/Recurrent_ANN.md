# Sequence data

Text speech, stock data. 

## Time series

Una qualsiasi misurazione presa in maniera periodica. 

- previsione numero di passengeri
- stock market data
- previsioni temporali, butterfly effect (piccoli errori nei computer possono portare a degli errori nelle previsioni meteorologiche)
- speech for audio, speech recognition
- text

### bag of words model

La mail è spam? Si crea un vettore di feature, e per ogni parola (nel dizionario) si vede quante volte occorre, e si aggiorna il contatore nel vettore. 

Si aggiunge quindi una feature finale, booleana "spam?".

Così, però, si perde il valore sull'ordine delle parole. 

Questa sezione gestirà soltanto sequenze di dati.


## Cos'è una sequenza?

Per una sequenza temporale, quale sarà la sua shape?

Non vogliamo che ogni punto sia un sample, nè che sia una feature.

Usiamo T per indicare la lunghezza della sequenza.

$$N\times T \times D$$

Dove 

N := numero di sample; 

D := numero di feature; 

T := numero di time stamp

e.g. 

modelliamo il percorcorso di una persona per andare al lavoro

D:= sarà pari a 2, poiché il gps restituisce lat e long
N:= il numero di percorsi separati (ovvero, in giorni diversi vi saranno percorsi diversi, ognuno di questi rappresenta un elemento a sé stante)
T:= il tempo del viaggio, tenendo conto della periodicità del campionamento

Per ora, supponiamo che ogni viaggio abbia la stessa lunghezza.


Nel caso dello stock price. 

D=1 (il valore dello stock)

T= potrebbe essere 10

N:= il numero totale di finestre, sulle quali effettuare il training.

In generale, avendo una sequenza di lunghezza L e ogni finestra è di dimensione T, avremo 

$$ N=L-T+1$$

Tuttavia, molto spesso vi sono più di uno stock da considerare alla volta, D>1. 


---



$$$$









