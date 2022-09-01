# Introduction

- Input image
- filter (kernel)
- output image (ottenuta come convoluzione dell'input con il filtro)

La convoluzione è indicata con il l'asterisco (*)

Esempi:
* il blur
* edge detection


Può essere vista come una feature transformation (che dipende dal filtro usato).

## Come funziona?

Un'immagine è una matrice di interi, un filtro sarà anche una matrice, più piccola. 

Facciamo una element wise multiplication e sommiamo tra loro i risultati.

N:=input length

K:=kernel length

Output_length:=N-K+1


La formula è
$$(A*w)_{ij}=\sum_{i'=0}^{K-1} \sum_{j'=0}^{K-1}A(i+i',j+j')w(i',j')$$


E' una operazione commutativa.

In DL si usa la cross-correlation. 


        convolve2d(A,np.fliplr(np.flipud(w)), mode='valid)

il movimento del filtro è legato agli estremi dell'immagine, quindi, l'output sarà sempre più piccolo dell'input. 


## Padding

E se volessi che l'input fosse della stessa dimensione dell'output? Si usa il padding, ovvero, si aggiunge un array di zeri intorno all'input (virtualmente). 

Tuttavia, anche col padding si perdono delle info, perciò, si aggiunge un'altra corona di zeri intorno all'input (non molto usata oggi).

<br>

| Mode | output size | usage |
|:---|:---|:---:|
|valid | N-K+1| typical|
|same | N | typical|
|full | N+K-1| atypical|
|

# A new perspective

## Vectorization

$$ a^T b =  \sum_{i=1}^{N}a_ib_i$$

Il prodotto scalare è anche:

$$a \cdot b= \sum_{i}a_i b_i = |a||b|\cos{\Theta_{ab}}$$

Anche detta similarità del coseno. In pratica, rappresenta la proiezione di un vettore sull'altro. Se due vettori sono paralleli, la similarità è massima, se perpendicolari è minima. 

$$\cos{\Theta_{ab}}=\frac{a\cdot b}{|a||b|}$$

Quindi, se abbiamo due vettori

$$cos(a,b)=\frac{\sum_{i=1}^{N}a_ib_i}{\sqrt{\sum_{i}a_i^2}\sqrt{\sum_{i}b_i^2}}=\frac{a \cdot b}{|a||b|}$$


La correlazione di Pearson 

$$\varrho_{ab}=\frac{\sum_{i=1}^{N}(a_i-\bar{a})(b_i-\bar{b})}{\sqrt{\sum_{i}(a_i-\bar{a})^2}\sqrt{\sum_{i}(b_i-\bar{b})^2}}$$

E' molto simile al coseno. Per questo motivo, la convoluzione è molto simile alla correlazione. 

Il prodotto scalare è una misura di correlazione, valori elevati indicano elevata correlazione positiva ecc...

## I filtri come pattern finder

Un filtro non è altro che un pattern finder. Difatti, la convoluzione va a azzerare tutto ciò che non è correlato con il filtro. 

e.g. 

un filtro per gli occhi, metterà in evidenza solo la parte dell'immagine contenente degli occhi. (uno sliding pattern finder)


 ## La convoluzione come matrix multiplication

 Possiamo trasformare la convoluzione in un prodotto matriciale. Creando una matrice dal vettore di partenza, ciò, tuttavia, porta a un aumento dello spazio occupato.

  E se invece facessimo il contrario? Usando la convoluzione al posto del prodotto matriciale

  $$ a=W^T x$$

Se nella matrice W vi fossero solo due pesi ripetuti più e più volte, potremmo risparmiare molto in termini di performance. 

Solitamente, le immagini sono di dimensioni troppo elevate. 

---

### Translation invariance

---

In alcuni casi, non serve usare le reti totalmente connesse. Nel caso delle immagini, un oggetto potrebbe semplicemente spostarsi. Nel caso di reti totalmente connesse i pesi dovrebbero essere allenati separatamente, e ciò dovrebbe accadere per ogni posizione. Conviene usare uno shared pattern finder che guardi a tutte le posizioni. Non serve imparare il concetto di gatto specificatamente per ogni particolare posizione. 

---

# Convolution on color images

Nel caso delle immagini colorate, abbiamo un parallelepipedo, e anche il filtro è un cubo


$$ (A*w)_{ij}=\sum_{c=1}^{3}\sum_{i'=1}^{K} \sum_{j'=1}^{K}A(i+i',j+j',c)w(i',j',c)$$

Tuttavia, così l'output sarebbe non tridimensionale, ma bidimensionale. Il che ci porta a pensare che non potremmo mettere un altro livello al livello successivo.

L'idea, è che idealmente avremo più filtri, in quanto vogliamo trovare più features. 

Avremmo più output, cosa ne facciamo? li mettiamo l'uno sopra l'altro, ottenendo una strutttura tridimensionale (possiamo mettere quanti livelli vogliamo).

Come vettorizzare su più filtri? 

## Vectorization

$$shape(A)=H \times W \times C_1$$
$$shape(w)=C_1 \times K \times K \times C_2$$
$$shape(B)=H\times W \times C_2$$

C2 dipende dal numero di feature, non è realemnte un colore. 

$$ B(i,j,c)=\sum_{i'=1}^{K} \sum_{j'=1}^{K}\sum_{c=1}^{C_1} A(i+i',j+j',c)w(i',j',c)$$


<br>

## Features map

E' corretto definire i C_2 dei colori? Poiché cerchiamo delle features nelle immagini, queste possono essere definite "feature map". L'output finale è la sovrapposizione dei vari singoli output.


---

## Convolution layer

Si aggiunge un bias e una funzione di attivazione. 

---

### Bias

La dimensione di b non sarà pari a $W*x$. Il vettore b è di dimensiione C2, si aggiunge lo stesso valore per ogni feature map.


---

# Architettura

Una rete è una serie del tipo

conv -> pool -> conv -> pool -> .... -> dense -> dense

I primi livelli trovano le feature, e poi la rete esegue della classificazione non lineare


## Pooling layer

Fa downsampling. In generale, si riduce la dimensione dell'immagine.

Vi sono due tipi di pooling:



1. Max: si suddivide la matrice in 2x2 square e ritorna il valore massimo.

2.  average: si suddivide e si prende la media dei valori


Perché?
- l'immagien è più piccola e semplice da gestire
- translational invariance (non mi interessa dove la feature appare, l'importante è che accada).

Prendendo il massimo mi concentro sulle zone dove il pattern è presente. L'avg fa qualcosa di simile.

Si possono avere box non di dimensione 2x2, solitamente non si fa così. Inoltre, questi box potrebbero sovrapporsi, ma anche questo non è comune.

## Perché la coppia conv-pooling?

Le cnn possono imparare feature in maniera gerarchica (prima i dettagli, poi aspetti più complessi). 

Ad ogni conv-pool l'immagine diventa più piccola. Si noti che, all'inizio, il filtro è reltaivamente più piccolo dell'immagine, più si va avanti, più questo aumenta rispetto all'immagine, che nel frattempo sta diventando più piccola. 

- l'input diventa piccolo
- il filtro cerca per feature sempre più piccole (gestione gerarchica delle features)


## Losing information? 
Perdiamo informazioni spaziali. Non ci interessa dove l'ogetto si trovi, l'importante è che ci sia l'oggetto.

Il numero di feature map tende ad aumentare al procedere. 

Si perdono spatial info, ma si guadagnano features.

## Hyperparameters

- Filter size
- feature map
- pool size

Vi è un pattern da seguire

1. small filter rispetto all'immagine 2x2 3x3 5x5 7x7
2. conv pooling sequences
3. incrementa il numero di features map ad ogni livello: 32 64 128 128 ... 
   
Leggere tanti papers. 


## Stride

Un'alternativa al pooling è lo stride. Quanto distanti deve stare ogni finestra. 

In pratica, si può saltare il pooling effettuando una convoluzione di tipo stride, come avevamo discusso in precedenza.

Ogni pixel, molto probabilmente, avrà lo stesso valore di quello vicino. i.e. pixel vicini sono correlati. 

Avere striding di 2 vuol dire saltare quei pixel. 

Quindi possivamo avere:

- conv pool con filter size uguali e aumento di features map
- strided conv con stesso filter size e aumento di features map


## Dense ANN part

Per accettare l'immagine, dobbiamo usare flatten(). 

Un'alternativa p il global max pooling layer

### Global max pooling

Le immagini non sono solitamente della stessa dimensione. 

Perché solo flatten non va bene? Genera input di dimensione diversa.

GMP: se l'immagine ha dimensione:
$$ H \times W \times C$$

L'output sarà un vettore di dimensione C. Ovvero, prende il massimo su ogni feature map. Usa l'idea del pooling: non mi serve sapere dove si trova la feature nell'immagine, basta che da qualche parte sia; qui la feature può trovarsi dappertutto su tutta la feature map. Perciò, può gestire qualsiasi immagine, di qualsiasi dimensione. 

Esiste la versione che usa l'avg al posto del max.

Tuttavia, con immagini troppo piccole, questo genera degli errori (ovvero, giunti al 2x2 non si può più ridurre la dimensione dell'immagine).

La gestione del dense, dipende dal task che vogliamo risolvere.

---

# Code preparation


- caricare i dati: fashion mnist (28x28 grayscale) and cifar-10 (color images 32x32)
- build the model (functional api)
- train
- evaluate
- prediction

        tf.keras.datasets.fashion_mnist.load_data()

        tf.keras.datasets.cifar10.load_data()

        train and test <= load_data()

fashion: Nx28x28 grayscale

Serve aggiungere una dimensione in più di dimensone 1, Nx28x28X1

CIFAR ha label di dimensione nx1, useremo flatten.


## Functional API

Nel modello di base si crea il modello aggiungendo argomenti nel costruttore.

Nella versione funzionale

                i = Input(shape(D,))
                x=Dense(128,activation...)(i)
                x=Dense(K,activation=...)(x)

                model=Model(i,x)

                ...
                model.fit(...)
                model.predict(...)



E' possibile creare modelli con più input e output

                model=Model(inputs=[i1,i2,i3],outputs=[o1,o2,o3])


Conv2D

        Conv2D(#output feature maps 32,
                (3,3) # filter dimension (special dimension solo qualcosa del tipo 2x2 5x5 ecc...)
                strides=2, #la velocità del passaggio sull'immagine
                activation
                padding=(default)'valid' 
                'same' 'full')

---

## Dropout, si o no? 
Si può aggiungere dropout regularization? si ma meglio non farlo. Si eliminano pixel. 



# Data augmentation

Si creano on the fly le versioni modificate delle immagini. Si prende un batch e si fa la versione augmented.


                from tensorflow.keras.preprocessing.image import ImageDataGenerator

                data_generator=ImageDataGenerator(
                        width_shift_range=0.1,
                        heigth_shift_range=0.1,
                        horizontal_flip=True
                )

Altri argomenti sono

        rotation_range, width_shift_range, heigth_shift_range, brighteness_range, shear_range, zoom_range, horizontal_flip, vertical_flip


Shearing è come tearing apart.

Non sempre il flip è possibile, 6 e 9 sono cose diverse.


Il prossimo step è flow

        data_generator= ImageDataGenerator(...)
        train_generator=data_generator.flow(
                x_train,y_train,batch_size
        )

        steps_per_epoch = x_train.shape[0] // batch size

        r = model.fit_generator(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epoch=50
        )



# Dati non normalizzati

Come rendere i dati normalizzati per ogni layer? Potremmo inserire un layer che effettua la normalizzazione del batch

batch norm -> dense -> batch norm -> ...

il batch norm fa la normalizzazione e la modifica tramite parametri che vengono sistemati automaticamente. 

La batch norm riesce a effettuare la regularization. E' come se si aggiungesse del rumore, delle immagini con rumore, e perciò, diviene resistente al rumore.

La batch norm non è solitamente usata trai dense, ma trai conv-conv e conv-flatten. 

VGG RESNET

batch normalization paper







   














