# Perché usiamo le ANN?
Molto spesso, lo spazio dei dati non è linearmente separabile. Un modo per risolvere è tramite la feature engeneering. Le possibilità di trasformazione crescono molto velocemente all'aumentare del numero di feature.

        playground.tensorflow.org

La presenza della funzione di attivazione rende l'intera rete non lineare. Altrimenti, potremmo ridurre tutta la rete in una sola equazione lineare.

# Activation functions

## Sigmoid e tanh

$$ \sigma(a) = \frac{1}{1+exp(-a)}$$


Il problema della sigmoid è che i suoi valori sono tra 0 e q, il suo output non potrà mai essere centrato su 0. Il problema è che l'output di un layer diviene l'input di un altro, e noi vorremmo che tutti gli input fossero standardizzati. 

Una soluzione è la tanh

$$ tanh(a) =\frac{exp(2a)-1}{exp(2a)+1}$$

E' una sigmoide, però centrata in 0. 

$$ tanh(a) \in [-1,1]$$


Dimostriamo che la tanh è solo una versione scalata e spostata verticalmente della sigmoide

$$ \sigma(x)=\frac{1}{1+e^{-x}} $$
Scaliamo e shiftiamo moltiplicando per 2 e sottraendo di 1

$$\frac{2}{1+e^{-x}}-1 = \frac{1-e^{-x}}{1+e^{-x}}$$

Poiché si tratta di una funzione dispari vale che $f(-x)=-f(x)$ quindi:

$$ -\frac{1-e^x}{1+e^x} = \frac{e^x-1}{e^x+1} =tanh(x)$$

Come volevasi dimostrare


### Problemi (vanishing gradient problem)
sigmoid e tanh sono derivabili, tuttavia, non funzionano molto bene.
Il problema della scomparsa del gradiente.
Quando si ha una rete profonda, il gradiente deve essere moltiplicato all'indietro. Più si va indietro, più termini ci sono nella chain rule, tuttavia, la derivata della sigmoide ha un massimo tra 0.2 e 0.3 (0.25), se moltiplichiamo tali valori, più termini vi saranno più il valore finale si avvicinerà a zero. Ciò farà sì che, nei livelli più indietro nella rete, non vi saranno variazioni significative dei pesi e bias. 


La soluzione è stata, usare la ReLU.

---

## ReLU

$$R(z)=max(0,z)$$

Il valore pari a zero potrebbe portare al problema del dead neuron. Tuttavia, funziona in pratica? Si

## ELU

$$f(x)= x>0 ? x : a(e^x-1)$$

Permette all'output di essere negativo (si avvicina all'idea che la media dei valori sia vicina a zero)

## Softplu

$$f(x)=log(1+e^x)$$

Per valori di x molto grandi è lineare. 


Tuttavia, questi ultimi esempi, mostrano che le funzioni non sono centrate in zero (ReLU)

<hr>

## Perché usare ReLU?

Usare le alternative non serve spesso. Si usa principalmente ReLU. Inoltre, questa ha un comportamento più simile ai neuroni reali, frequency coding.

### Similarità biologica

Più l'input è forte, più lo saraà l'attivazione. Come la relu, inoltre, non ci possono essere frequenze negative.

        Da approfondire: BRU (bionodal root unit)


Useremo la Relu negli hidden layers. 
---

# Multiclass classification

Nel caso in cui vi sono k possibili uscite, serviranno k neuroni in uscita, corrispondente ad una sorta di probabilità.

Dovranno essere:
- non negativi
- $\in [0,1]$
- la loro somma deve essere, complessivamente, pari a 1

## Softmax

 $$p(y=k|x)=\frac{exp(a_k)}{\sum_{j=1}^{K}exp(a_j)}$$

 - Sempre positivo
 - la somma da 1 poiché al num e al den otteniamo sempre lo stesso valore

        tf.keras.layers.Dense(K,activation='softmax')

Non usata per gli hidden layer. 

La softmax è più generale, può gestire anche la classificazione binaria, basta porre K=2, basterebbe un solo neurone in output.

---

# Come rappresentare le immagini

Le immagini su un computer sono rappresentate come pixel. Come una specie di matrice. 

Come rappresentare i colori? 

RGB sono colori primari, che possono essere usati per generare qualsiasi altro colore. 
Ogni elemento della matrice è una tripla, che indica la quantità di R G B.

Avremo un tensore A(i,j,k).

8 bit sono sufficienti, $2^8=256$ valori possibili.

Se un'immagine ha dimensione 500x500 
, ci serviranno:

$$500 \cdot 500 \cdot 3 \cdot 8$$

Se le immagini sono in bianco e nero è più semplice. ci basta 2d arrays. 

Con imshow:

        plt.imshow(array2s, cma='gray')

per le immagini in bianco e nero.

## Nelle ANN
Per le ANN conviene scalare le immagini, in modo tale da renderli valori float tra 0 e 1. Ma non sono centrati! Si tratta di un'eccezione 

## VGG

Per questa ANN gli input non sono scalati, ma sono nel range 256.

## Come rappresentare l'immagine come input della ANN?

Trasformiamo il tensore in un array, tramite la funzione:

        flatten() 

---

# Struttura del codice

- carica i dati
- costruire il modello
- allena il modello (fit)
- valuta il modello
- esegui delle predizioni

        tf.keras.datasets

## MNIST

train_X = Nx28x28

train_Y = N

test_X  = N_test x 28 x 28

test_Y  = N_test

Il valore dei pixel sarà [0,255] da dover scalare

### Struttura rete

Flatten permette di passare una matrice. 
Il primo livello sarà di 128. 

Dropout (regolarizzare) evita di dipendere da alcuni input. Droppa qualche input, randomicamente per ogni esecuzione. Vi è una probabilità del 20% di droppare un nodo ogni qual volta si va attraverso la rete (i.e. mettere a zero il nodo).

L'ultimo livello avrà softmax. 


L'ottimizzatore sarà adam, e sparse_categorical_corssentropy (da approfondire)
e metrics accuracy.

---

### Cross entropy loss

Per un singolo sample.

$$Loss=-\sum_{k=1}^{K}y_k^{(OH)}\log{\hat{y_k}}$$

$$where : \hat{y_k}=p(y=k|x)$$

Mentre OH indica il one hot encoding. 9 => [0,0,0,0,0,0,0,0,0,1]

Se la nostra predizione fosse perfetta, 

$y=k \rightarrow y_k^{OH}=1$ and $\hat{y_k}=1$

So
$$-1*log1=0$$

Se la predizione fosse la più sbagliata possibile, invece:
$y=k \rightarrow y_k^{OH}=1$ and $\hat{y_k}=0$

Quindi

$$-1*\log0=\infty$$


$\textbf{Perché sparse-cross-entropy?}$

Per ottenere la cross-entropy, sia la y_OH che Y_target devono essere array di dimensione K. Il che è sub ottimo, in quanto il target potrebbe essere un semplice intero (come in mnist)! Ci servirebbe un array di soli zeri e un solo 1.

Sparse indica che vi è un array di moltissimi zeri, che non contribuiscono per nulla alla funzione di loss.

Difatti, definiamo:

$$Loss=-\log{(\hat{y}[k^*])}$$

Ci basta considerare solo la k-esima entry della prediction della rete, in quanto le altre saranno annullate (la restante parte del vettore OH target conterrà infatti solo zeri).

Non ci serve neanche fare il one hot encoding del vettore target.






























