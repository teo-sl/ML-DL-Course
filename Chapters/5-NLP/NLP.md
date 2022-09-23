# Legami con le RNN

Nel caso delle parole, queste non sono contigue come nel caso dei numeri.

Per gestire questo aspetto si potrebbe fare one hot encoding. Per trasormare da categorico a numerico. 

i.e. avremo una mappa di vettori, dove un solo elemento è diverso da zero. 

Se abbiamo una sequenza di T parole, e ognuna è un vettore di D elementi, possiamo ricondurci al caso TxD, che può essere inviato in input alla RNN.

Questo approccio genera problemi, il valore di D può essere troppo grande. Il numero totale di parole può essere eccessivamente grande.

Effettuando il one hot encoding, la distanza euclidea trai vari oggetti non ha alcun legame particolare con il significato, la distanza sarà sempre pari a $\sqrt{2}$ .

Quel che si fa è assegnare a ogni parole a un array d-dimensionale "EMBEDDINGS"


Prendere un vettore one hot encoded e moltiplicarlo per una matrice non fa altro che selezionare una riga della matrice dei pesi. Quello che possiamo fare, invece di effettuare la reale moltiplicazione è indicizzare la matrice utilizzando il vettore OHE.

## Embedding layer

Possiamo semplicemente mappare le parole a interi (che indicano l'indice della matrice) e poi dare in pasto l'input originale al layer di embedding.

        [i, like, cats] -> [50,25,3]

Una volta mappate le parole in indici, si prendono i vettori corrispondenti 

        [50,25,3] -> [[...],[...],[...]]

Ottenendo così una matrice TxD.


## Quale criterio usare?

L'idea di fondo è che parole simili dovrebbero essere vicine. Dal momento che si tratta di pesi (i vettori sono pesi), questi vengono allenati automaticamente dalla rete. 

Spesso si usano pre-trained vectors. 

# Code preparation

Ogni parola deve essere trasformata in un intero.


        dataset = sequences of word
        current_idx = 1
        word2idx={}
        for word in dataset:
            if word not in word2idx:
                word2idx[current_idx] = word
                current_idx++

Si parte da 1 perché, Se l'input è del tipo NxTxD, dove T rappresenta la frase di lunghezza massima, qualsiasi frase di lunghezza inferiore dovrà avere del padding, che verrà segnalato con 0.

In tensorflow, vi è una funzione che fa tuttto in maniera automatica. Difatti, un testo è una singola stringa, è necessario separare le parole in una sequenza!

Ci serve una lista di stringhe, tale processo è detto tokenization. 

string $\rightarrow$ tokens $\rightarrow$ integers $\rightarrow$ vectors


Il tokenizer di tensorflow fa tutto questo in modo automatico.

        MAX_VOCAB_SIZE=20000

        tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE) # utile per evitare le parole rare o mispelling, et similia
        tokenizer.fit_on_text(sentences)
        sequences = tokenizer.texts_to_sequences(sentences)


Ogni frase ha lunghezza diversa, per evitare questo, è necessario far uso del padding. 

In tensoflow viene fatto da una funzione.

        data = pad_sequences(sequences,maxlen=MAXLEN)

Di default, maxlen è None, e si adatta alla frase più lunga, tuttavia, è possibile troncare dove necessario. 

Se si deve troncare, si può scegliere se troncare all'inizio o alla fine

        truncating="pre" o "post"

        padding="pre" o "post"

Possiamo anche mettere il padding alla fine o all'inizio, con l'opzione 'padding'. 

                i = Input(shape=(T,))
                x=Embedding(V,D)(i) # dove v è l'indice


                i = Input(shape=(T,))
                x=Embedding(V,D)(i)
                x=LSTM(x)
                x=GlobalMaxPooling1D()(x)
                x=Dense(K,activation='sfotmax')(x)

Dove 

T = sequence length

D = embedding dimensionality (si può scegliere)

M = hidden vector (si può scegliere)

K = numero di classi in output


# CNN for NLP

Come per le immagini, dove l'idea fondamentale è che, pixel vicini hanno anche valori simili, anche per le sequenze, valori successivi hanno valori simili. La convoluzione potrebbe essere utili anche qui. La 1D convolution è più semplice di quella in 2D.

Parliamo di cross-correlation


$$x(t)*w(t)=\sum_\tau x(t+\tau)w(\tau)$$

Possiamo aggiungere features


$$y(t,m)=x(t)*w(t)=\sum_\tau \sum_{d=1}^{D} x(t+\tau,d)w(\tau,d,m)$$

m è il numero di output features. 

Perciò:

- l'input è TxD (T il numero di time steps, D il numero di input features)
- l'output è TxM (M è il numero di output features)
- W è il filtro, e ha dimensione T' x D x M (dove T' << T)

Per le immagini:
- due dimensioni spaziali + 1 dimensione delle input features + 1 dimensione per le output features = 4

Per le sequenze:
- 1 dimensione per il tempo + 1 dimensione per le input feature + 1 dimensione per le output feature = 3


Come applichiamo al testo? Difatto, è quello che otteniamo con l'embedding layer, dopo il vettore di numeri che corrisponde all'indice delle parole, otteniamo un vettore per ogni parola, e quindi, una matrice di vettori.

                i = Input(shape=(T,))
                x = Embedding(V+1,D)(i) # output TxD
                x=Conv1D(32,3,activation='relu')(x) # output T' x M
                x=MaxPooling1D(3)(x)
                Conv1D(64,3,activation='relu')(x) # output T2 x M2
                x=MaxPooling1D(3)(x)
                Conv1D(128,3,activation='relu')(x)
                x=GlobalMaxPooling1D()(x) # output M3
                Dense(1,activation='sigmoid)



