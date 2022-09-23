# Introduzione

Il principale obiettivo è generare dati, principalmente immagini. 

 Il sistema è costituito da due ANN:

 1. generare immagini
 2. discriminare tra immagini reali e immagini generate


Si parla di adversarial perché le due reti hanno degli obiettivi in contrasto.

Qual è la loss function(s)?

Ne avremo due, una per il generator e una per il discriminator.

Il discriminator ha un problema binario (binary crossentropy), vero falso.


        generator -> fake image -> discriminator -> 1/0


Blocchiamo i pesi nel discriminator (alleniamo solo il generator), continuando a usare la binary classification, semplicemente, cambiando le label. Passeremo immagini false ma le etichetteremo come vere.


$$J_G = -\frac{1}{N} \sum_{n=1}^{N} log{(\hat{y_n})}$$
$$\hat{y_n}=fake_image,target_is_always_1$$

Ci interessano solo quelli pari a uno. Stiamo allenando il generatore.


Nel caso del discriminator, l'input è una immagine, per il generator?

E' soltando del noise. Il generator imparerà a tradurre quel rumore (normale) in delle immagini. (latent space) E' uno spazio dove la rete immagina vivano tutte le immagini che andrà a generare (vedi latent space variation). 

<img src="https://miro.medium.com/max/793/1*RkCl-6gbVql7WP86yjUclQ.png" />

Il generator riesce a mappare in modo continuo i valori in un'immagine.

Un generator è un reverso del feature transformer o embedding. Questi partono dall'ìimmagine e la mappano in un vettore di interi, qui avviene il contrario.


## Passi


        # si carica il dataset
        x,y=get_mnist()

        # si creano i modelli 
        d=Model(image,prediction)
        d.compils(loss='binary_crossentropy')
        g=Model(noise,image)


        # si combinano i modelli
        fake_prediction=d(g(noise))

        combined_model=Model(noise,fake_prediction)
        combined_model.compile(
            loss='binary_crossentropy',...
        )

        #sgd loop

        for epoch in epochs:
            real_images = sample from x
            fake_images = g.predict(noise)
            d.train_on_batch(tral_images,ones)
            d.train_on_batch(fake_images,zeros)

            # train generator
            combined_model.train_on_batch(noise,ones)


## Un appunto

Nel caso in particolare, vogliamo loss down e accuracy high? In questo caso, dovremmo avere del rumore, ovvero, un andamento up and down, d e g sono in competizione!
