# Introduction

L'idea è prendere un modello già allenato e usarlo per i nostri task. L'idea è che i modelli condividano aspetti.

Di fondo, si mantiene la CNN e si cambia la testa, i.e. la rete neurale, si allena solo la parte in testa, mentenendo fissi i pesi nel libello convoluzionale. 

Non servono troppi dati, in quanto la CNN è già stata allenata.

# Alcuni esempi

- VGG: Visual Geometry Group. Anche VGG16 e VGG19

- ResNet: è gestita a branches. Esistono variazioni.

- Inception: simile a resnet per i branch paralleli. Qui vi sono più convoluzioni in parallelo. Si provano tutti i filtri (di diverse dimensioni) e si concatenano.


Quale dovrei scegliere? Bisogna provare! Non vi è modo di predirre. 

- MobileNet: è lightweight, ma non è il meglio per performance. Utile per i mobile e embedded devices.


L'input data deve essere nello stesso formato degli originali.

        from keras.applications.resnte50 import ResNet50, preprocess_input

Di solito si lavora con pixel values in [0,1] o [-1,+1].

VGG usa pixel centrati ma non scalati.

Non serve preoccuparsi di ciò, vi è una funzione preprocess_input che si occupa di tale aspetto. 


ResNet 224x224 


# Large Image Datasets

Tutte le immagini non entrerebbero in RAM. 

Usando batch gradient descent, carichiamo in memoria solo il batch di interesse!

Alla prossima iterazione caricheremo il prossimo batch 

        gen = ImageDataGenerator()

        generator = gen.flow_from_directory()

Ciò ci permette di effettuare un resize di 224x224

        model.fit_generator(generator)

In questo modo, prenderà i dati e effettuerà il resize.

La strutttura delle directory dovrà essere

        \data\train\k1
        \data\train\k2
        ...
        \data\train\k_n
        

        \data\validation\k1
        \data\validation\k2
        ...
        \data\validation\k_n

Dove n è il numero di classi.


# Approcci possibili

Possiamo precalcolare tutti i valori in output della CNN, evitando di doverlo fare dopo. Il problema, tuttavia è che con data augmentation, l'immagine viene modificata di volta in volta, quindi, z sarà diversa per ogni iterazione, non possiamo perciò effettuare data augmentation.

1. usa image data generation nel loop
2. calcola a priori il valore del vetttore Z senza data augmentation. 


Naturalmente, 1. è lento rispetto a 2.

Tuttavia, 2. è meno generale, quindi ci restituisce un modello che non riesce a generalizzare. 





