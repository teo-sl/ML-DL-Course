# Struttura

        (user, item, rarting) 

Questi dataset saranno incompleti, e devono esserlo, i.e., devono esserci item che l'utente x non ha usato/visto ecc... tali che il sistema li consigli all'utente.

# Come si consiglia?

il modello è tale che

$$f(u,m) \rightarrow r$$

dato un item e un utente, riesce a predirre al meglio il punteggio.

Si prendono tutti gli item non recensiti e si prende quello con rating più alto.

## Encoding degli item

Usiamo un embedding, ogni elemento viene mappato su un vector di features. 

Si prende il feature vector dell'utente e dell'item e si concatenano, il risultato viene dato in pasto a una ANN. 

Il task da risolvere è di regressione, dal momento che cerchiamo di predirre il punteggio


        u = Input(shape=(1,))

        m = Input(shape=(1,))


        u_emb = Embedding(num_users,embedding_dim)(u)

        m_emb=Embedding(num_movies, embedding_dim)(m)

        # le dimensioni non devono essere necessariamente uguali, ma perché no?

        x = Concat()(u_emb,m_emb)

        # ANN

        x = Dense(512,activation='relu')(x)
        x=Dense(1)(x)

L'uso della functional API è molto importante in questo caso. 