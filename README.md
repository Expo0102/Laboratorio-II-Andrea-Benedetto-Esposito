# Progetto Laboratorio 2
### Relazione progetto Laboratorio 2 - Andrea Benedetto Esposito (601201)

## Introduzione 
Il progetto consiste nella creazione di un programma in C che permetta la risoluzione del calcolo di pagerank su un grafo orientato, partendo da un file di input contenente i nodi e gli archi del grafo di tipo ".mtx". In particolare, il programma calcola il pagerank di ogni nodo del grafo in un contesto multithreaded e stampa a video i risultati. I
componenti principali del progetto sono:
- `pagerank.c` -> Cuore del progetto, contiene la logica principale del calcolo del pagerank e della creazione del grafo  
- `graph_server.py` -> File che contiene il server: si occupa di ricevere i dati dal client e di inviare i risultati, prende come argomento una serie di file ".mtx" contenenti una serie di archi e per ogni file esegue il calcolo del pagerank con un sottoprocesso
- `graph_client.py` -> File che contiene il client: si occupa di inviare i dati al server e di ricevere i risultati

## Scelte implementative

### Pagerank
Pagerank è il processo (o sottoprocesso lanciato dal server) che si occupa della creazione del grafo e del calcolo del pagerank di ogni nodo.
Prende in input il file ".mtx" contenente il grafo e una serie di parametri opzionali relativi al calcolo del pagerank che, se non specificati, assumono valori di default. 
Il grafo viene creato seguendo il paradigma produttore-consumatore.
- `produce` -> Funzione associata al thread producer che si occupa di leggere il file ".mtx" e di inizializzare il grafo; in particolare, inizializza una hashmap, legge il file riga per riga e per ogni riga crea un arco e lo inserisce nell'hashmap (per evitare duplicati), se l'arco non è già presente lo aggiunge nella coda condivisa lanciando la funzione `enqueue`.
- `thread_function_graph` -> Funzione associata ai thread ausiliari consumatori che si occupa di estrarre un arco dalla coda condivisa e di aggiungerlo al grafo; in particolare, estrae un arco dalla coda lanciando la funzione `dequeue` e se l'arco non è già presente lo aggiunge al grafo.
Dopo aver atteso che tutti i thread abbiano finito di creare il grafo si può procedere al calcolo del pagerank.

Il calcolo del pagerank in parallelo avviene in due funzioni principali `pagerank` e `pagerank_thread`:
- `pagerank` -> Funzione chiamata dal main che si occupa di inizializzare il pagerank di ogni nodo, di creare e di avviare il signal__thread con la funzione associata `signal_function` (questa funzione mette in attesa il thread fino alla ricezione del segnale `SIGUSR1`; quando riceve il segnale stampa in stderr l'iterazione corrente e il nodo con il relativo pagerank attualmente più alto), di inizializzare la struttura dati utilizzata dai thread ausiliari, di distribuire il lavoro in modo equo ($\simeq \frac{\text{nodi-totali}}{\text{num-thread}}$ nodi da calcolare per ogni thread) e di lanciare i thread che si occupano del calcolo del pagerank
- `pagerank_thread` -> Funzione associata ai thread ausiliari che si occupa del calcolo del pagerank di un insieme di nodi; in particolare,, calcola un'iterazione del gruppo di nodi assegnato e attende che tutti i thread abbiano finito il calcolo dell'iterazione eseguendo la funzione  `barrier_wait`, che mette in attesa i thread e li sveglia solo quando l'ultimo thread esegue barrier_wait. Prima di iniziare l'iterazione successiva il thread0 si occupa di aggiornare le variabili globali. 

### Server
Il server è progettato per ricevere grafi da client, salvare i dati su file temporanei, eseguire un algoritmo di PageRank su questi grafi e restituire i risultati ai client. Il server è in grado di gestire connessioni multiple grazie all'utilizzo di thread.
- `socket` -> Il server utilizza il modulo `socket` per creare un socket TCP, che ascolta le connessioni in entrata e comunica con i client. Questo permette una comunicazione bidirezionale affidabile.
- `threading` -> Per ogni connessione client, il server avvia un nuovo thread tramite il modulo `threading`. Questo permette al server di gestire più connessioni client contemporaneamente senza bloccarsi.
- `logging` -> Il server utilizza il modulo `logging` per registrare eventi significativi, come la gestione di errori, in un file di log. Questo aiuta nella manutenzione e nel debugging.
- `signal` -> Il server gestisce il segnale SIGINT (ad esempio, generato dall'input Ctrl+C) per chiudere in modo pulito il socket del server e terminare il processo.
- `file temporanei` -> Per ogni connessione client, il server crea un file temporaneo dove scrive i dati del grafo ricevuti dal client. Questo file viene poi passato a un programma esterno per calcolare il PageRank.

### Client
Il client è progettato per inviare grafi al server e ricevere i risultati del calcolo di PageRank. Il client si connette al server tramite un socket TCP e invia i dati del grafo al server. Dopo aver inviato i dati, il client attende i risultati dal server e li stampa a video. 
- `socket` -> Il client utilizza il modulo `socket` per creare un socket TCP e connettersi al server. Questo permette una comunicazione bidirezionale affidabile.
- `logging` -> Il client utilizza il modulo `logging` per registrare eventi significativi, come gli errori, in un file di log. Questo aiuta nella manutenzione e nel debugging.

## Esecuzione
Allinterno della cartella è presente un makefile che permette di compilare `pagerank.c` con il comando `make`.
Per eseguire il programma è necessario lanciare il server e il client in due terminali differenti. 
- `python3 graph_server.py` -> Avvia il server, che si mette in ascolto di connessioni in entrata
- `python3 graph_client.py file1.mtx file2.mtx ...` -> Avvia il client, che si connette al server e invia i dati del/dei file ".mtx" come argomento.
- `./pakerank -k [num top] -t [num thread] -m [max iter] -e [max error] -d [bumpind factor] file.mtx`-> Esegue il calcolo del pagerank su un singolo file ".mtx" passato come argomento con i parametri opzionali specificati







  
