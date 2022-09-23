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

# Linear regression e autoregressive

E' possibile riprodurre la funzione seno con solo il valore e t-1 e t-2

$$x(t) = sin(\omega t)$$

$$sin(\omega(t+1))=w_1 sin(\omega t) + w_2 sin(\omega (t-1))$$

Moltiplichiamo il termine omega.

$$sin(\omega t + \omega) = w_1 sin(\omega t) + w_2 sin(\omega t - \omega)$$

Sappiamo che

$$sin(a+b)+ sin(a-b) = 2cos(b)sin(a)$$


Modifichaimo il termine di prima

$$sin(\omega t + \omega) - w_2 sin(\omega t - \omega) = w_1 sin(\omega t)$$

Da cui si evince che 

$a=\omega t$ e $b = \omega$

Mentre, per le costanti: $2cos(b)=w_1$ e $w_2-1$

Da cui deriva il fatto sia possibile rappresentare il seno con una relazione ricorrente (come la serie di fibbonacci)



# Reti neurali ricorrenti

Nel caso in cui D non è pari a uno, non possiamo usare l'approccio mostrato prima. 

Potremmo fare un flatten, concatenando le singole sequenze. Tuttavia, potrebbe non essere una buona idea. Avremmo un vettore in input troppo grande. 

Possiamo prendere ispirazione dal forecasting, dove usiamo i valori precedenti. 

Facciamo sì che i livelli precedenti prendano in loop i valori in output, in modo tale da usare i valori precedenti come input. 

$$h_t = \sigma (W_{xh}^T x_t + W_{hh}^T h_{t-1}+b_h)$$

Solitamente, si usa un solo hidden layer.

## Glossario

x = input

h = hidden

o = output

xh = input to hidden

hh = hidden to hidden

## Continuazione

$$h_t = \sigma (W_{xh}^T x_t + W_{hh}^T h_{t-1}+b_h)$$

$$\hat{y_t}=\sigma(W_o^T h_t + b_o)$$


Simple recurrent unit o Elman unit

## Come si calcola l'output? 

Sia data una sequenza in input

$$x_1, ... , x_T$$

Dove ogni elemento ha dimensione D

$$shapt(x_t) = D$$

al primo step, si calcola $h_1$, tuttavia, questo dipende da $h_0$, che non è stato calcolato. Solitamente, si mette un vettore di zeri, oppure, si può sceglere il valore ottimale usando la gradient descent. 

$$h_1 = \sigma (W_{xh}^T x_1 + W_{hh}^T h_{0}+b_h)$$

Avendo $h_1$ si può ottenere y

$$\hat{y_1}=\sigma(W_o^T h_1  + b_o)$$ 

Abbiamo ora $x_2$, che segue lo stesso procedimento. 

Continuiamo fino alla fine della sequenza.

Ogni $\hat{y_t}$ dipende solo da $x_1,...,x_t$

Che senso ha avere una y hat per ogni passo se stiamo facendo forecasting? Tutte le y hat, tranne l'ultima, sono scartate, 

Ci sono casi in cui invece ci servono le y hat intermedie. Un caso sono le neural machine translation.


## Classification probability

 Nel caso classico

 $$p(y=k|x)$$

 Nel caso delle RNN invece

 $$p(y_t = k | ?)$$


La dipendenza è la seguente

$$p(y_T = k | x_1,x_2,...,x_T)$$

Ovvero, la dipendenza è verso tutti gli input precedenti. 

### Markow models

La dipendenza di una parola è solo nei confronti della precedente. Non sono molto potenti. Mentre le RNN dipendono da tutti i valori.

Wxh - input to hidden weight

Whh - hidden to hidden weight

bh - hidden bias

Wo - hidden to output weight

bo - output bias

X - TxD input matrix

tanh hidden activation

softmax output activation
                

                Yhat=[]
                h_last = h0
                for t in range(T):
                        h_t = tanh(X[t].dot(Wx)+ h_last.dot(Wh)+bh)
                        yhat=softmax(h_t.dot(Wo)+bo)
                        Yhat.append(yhat)

                        h_last=h_t


Hopfield Network and Hebbian Learning 


# Code preparation

Stesso esempio di prima, al posto di autoregressive con RNN. 

1. load data (dataset generato sinteticamente)
2.  build the model
3.  train the model
4.  evaluate the model
5.  make prediction

1. t=0...len(series)-T a differenza di prima, abbiamo nxtx1 array, serve una dimensione superflua.
   
i=Input(shape(T,1))
x=SimpleRNN(5,activation='relu)(i)
x=Dense(1)(x)
model=Model(i,x)


loss='mse
optimizer=Adam(lr=0.1)

il resto è stimile al caso di prima


5. Se l'input è nxtxD, output NxK dove K è il numero di neuroni in output. Serve fare il reshape.

                model.predict(x.reshape(1,T,1))[0,0]

Per avere il risultato come uno scalare.

Togliendo l'activation function, si riduce a un linear model, perciò, la rete si comporta come un autoregressor. 


# Modern RNN units

LSTM (long short term memory)
GRU (gated recurrent unit), versione semplificata di LSTM

Perché ci servono queste versioni complesse delle RNN? 

Le rnn sono molto soggette al problema del vanishing gradiente, non riescono a imparare da fatti che stanno molto nel passato. 

Non si può usare ReLU, il modo più efficace è usare delle unità totalmente diverse (i.e. LSTM).

## GRU

Come per la SRNN vi è la dipendenza da x(t) e h(t-1).

z(t) update gate vector

r(t) reset gate vector

h(t) hidden state


Shapes: sono vettori di dimensione M, un iperparametro. 

$$h_t = (1-z_t)\bigodot h_{t-1} + z_t \bigodot tanh(W_{xh}^T x_t + W_{hh}^T(r_t \bigodot h_{t-1})+b_h)$$


$z_t$ indica se dobbiamo imparare il nuvo valore, o se fidarci del precedente.(è il risultato di una sigmoide, perciò appartiene a [0,1], quindi, il valore corrente di h_t è una media pesata di h_t e h_t-1).

z_t è una logistic regression (la sigmoide non è un iperparametro, è sempre così, per avere un valore tra 0 e 1), i.e. un neurone. E' una probabilità, prendiamo il nuovo valore o teniamo il vecchio?


$$h_t = p(keep h_{t-1})h_{t-1} + p(discard h_{t-1} SimpleRNN(x_t.h_{t-1}$$

In realtà non è esattamente così.

Il termine gate è rappresentativo. Questo è perché il cancello scorre tra l'una o l'altro valore, si dice convex combination.

<hr>

Cosa indica $r_t$? E' un semplice neurone. 

$$r_t = \sigma(W_{xr}^T x_t + W_{hr}^T h_{t-1}+b_r$$

Tale valore viene usato come element-wise multiplication a $h_{t-1}$.

r_t viene usato per capire quali parti di h_t-1 ricordare e quali dimenticare! Rafforza soltanto il concetto visto prima. 

A differenza di una normale RNN ha la capacità di ricordare e dimenticare. Si usano binary logistic regression per realizzare tali funzionalità.  

# LSTM

L'LSTM ha tre input, x(t) h(t-1) e c(t-1), l'ultimo indica il cell state, che, insieme a h(t) rappresenta l'output della cella (c(t)). 

Sono quindi necessari anche due stati iniziali $c_o$ e $h_o$.

La formula per il funzionamento è questa.

$$h_t = o_t \bigodot f_h(c_t)$$

Ognuna delle componenti delle formule è, in pratica, un neurone, i.e. un logistic binary classifier.

Ci sono più gate:

- $f_t$ il forget gate
- $i_t$ input/update gate
- $o_t$ output gate
- $c_t$ cell state
  
Il forget gate ci dice quanto del precedente valore c_t dimenticare

$$f_h$$

è un'altra funzione di attivazione, solitamente è una tanh.

In tensorflow non si può scegliere $f_c,f_h$ separatamente.

In sintesi:

- f(t) neuron, binary classifier
- i(t) neuron, binary classifier
- o(t) neuron, binary classifier
- c(t)=f(t) * c(t-1)+i(t)*SimpleRNN
- h(t)=o(t)*tanh(c(t))

## Code

                i=Input(shape=(T,D))
                x=LSTM(M)(i)
                x=Dense(K)(x)
                model=Model(i,x)

## Options per RNN units

Per ottenere tutti i valori precedenti, si deve aggiungere:

                x = LSTM(M,return_sequences=True)(i)

Vi è anche l'argomento:

                return_state=True

Nel caso di GRU non aggiunge nulla, nel caso di LSTM restituisce anche $c_T$


## global max pooling 1d

global max pooling 2d : HxWxC => C
global max pooling 1d : TxM   => M


# Alternative way to forecast

In alcuni casi può avere senso effettuare delle previsioni con un singolo step, altre volte no. 








