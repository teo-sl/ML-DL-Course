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


- Neural interface: T=1000, N= # letters to think, D=# of electrode

Vi sono anche altre convenzioni, ma noi useremo questa. La convenzione comprende anche D alla fine. 

## Variable length sequences

Le frasi hanno lunghezza diversa, ad esempio. 

Vi sono inefficienze sia usando lunghezze costanti sia variabili. 

Quello che si fa è prendere la sequenza più lunga e riempire quelle meno lunghe con padding di zero. Il che è molto inefficiente, tuttavia, rende il codice più semplice. In tf si fa così.



# Forecasting

Predire valori futuri multipli. 

Il modo più semplice per fare forecasting sembra essere la linear regression! Ma, linear regression attende valori NxD non NxTxD. Nel caso semplice D=1, quindi, abbiamo una dimensione superflua. Possiamo usare flatten().

Pretendiamo che T sia D.  

N sarà il numero di finestre di dimensione che stanno nella serie. 

Tra gli elementi di una finestra, l'ultimo rappresenta il nostro target. 

$$\hat{x_t}=w_0+w_1x_{t-1}+w_2x_{t-2}+w_3x_{t-3}$$

Viene detta autoregressive: cerca di predirre usando i suoi stessi valori.


NOn possiamo usare questa tecnica, perché vorremmo predirre molti step in avanti! Non solo uno. Dovremmo utilizzare i valori che abbiamo predetto per predirre i successivi! Le $\hat{x}$ entrano nella formula.

        x = last values of train set

        predictions = []

        for i in range(length_of_forecast):
        x_next = model.predict(x)
        predictions.append(x_next)
        x=concat(x[1:],x_next)

Scartiamo il valore più vecchio. 


        i = Input(shape(T,))
        x = Dense(10,activation='relu)(i)
        x = Dense(1)(x)

        model = Model(i,x)

Nel caso delle serie, non bisogna fare lo split random tra training e validation, si divide a metà invece.













