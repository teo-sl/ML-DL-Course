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











