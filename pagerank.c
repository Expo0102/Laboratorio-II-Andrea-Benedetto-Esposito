#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>
#include <getopt.h>
#include <signal.h>

// Srtuttura del nodo
typedef struct Node {
    int dest;
    struct Node* next;
} Node;

// Struttura del grafo
typedef struct {
    int N;
    int *out;
    Node **in;
    pthread_mutex_t mutex;
} grafo;

// Struttura dell'arco
typedef struct {
    int src;
    int dest;
} Arco;

// Struttura della coda di task
typedef struct {
    Arco *buffer;
    int size;
    int in;
    int out;
    sem_t empty;
    sem_t full;
    pthread_mutex_t mutex;
} TaskQueue;

// Struttura del thread pool
typedef struct {
    pthread_t *aux_threads;
    int num_aux_threads;
    TaskQueue *task_queue;
    grafo *g;
} ThreadPool;

// Struttura dei dati passati ai thread per la cosntruzione del grafo
typedef struct {
    grafo *g;
    TaskQueue *task_queue;
    int num_aux_threads;
    char *filename;
    ThreadPool *pool;
} ThreadData;

// Struttura del nodo dell'hashmap
typedef struct HashmapNode {
    int src;
    int dest;
    struct HashmapNode *next;
} HashmapNode;

// Struttura dell'hashmap
typedef struct {
    HashmapNode **table;
    int size;
} Hashmap;

// Struttura per salvare i deti del calcolo del pagerank
typedef struct {
    double *ranks;
    int iterations;
    double delta;
} PageRankResult;

// Struttura dei dati passati ai thread per il calcolo del pagerank
typedef struct {
    grafo *g;
    double *ranks;
    double *old_ranks;
    double damping;
    double *sum_dead_ends;
    int start;
    int end;
    double *local_deltas;
    int thread_id;
    int *iterations;
    double epsilon;
    int maxiter;
    int *stop_flag;
    int num_threads;
    double *delta;
} PagerankThreadData;

// Struttura dei dati passati al thread per la gestione dei segnali
typedef struct {
    int *iterations;
    double *ranks;
    int num_nodes;
    pthread_mutex_t *lock;
} SignalThreadData;


void initTaskQueue(TaskQueue *task_queue, int size);
void destroyTaskQueue(TaskQueue *task_queue);
void enqueue(ThreadPool *pool, Arco arco);
Arco dequeue(ThreadPool *pool);
void *thread_function_graph(void *arg);
void *produce(void *arg);
void initGraph(grafo *g, int N);
void addEdge(grafo *g, int src, int dest);
void freeGraph(grafo *g);
void threadPoolInit(ThreadPool *pool, int num_aux_threads, TaskQueue *task_queue, grafo *g);
void threadPoolDestroy(ThreadPool *pool);
Hashmap *createHashMap(int size);
void destroyHashmap(Hashmap *hashmap);
int hashFunction(int src, int dest, int size);
int hashmapContains(Hashmap *hashmap, int src, int dest);
void hashmapInsert(Hashmap *hashmap, int src, int dest);
PageRankResult pagerank(grafo *g, int num_nodes, double damping, double epsilon, int maxiter, int num_threads);
void *pagerank_thread(void *arg);
void *signal_function(void *arg);
int compare(const void *a, const void *b, void *ranks);
void initialize_sync();
void destroy_sync();
void barrier_wait(int num_threads);


// Variabili per la sincronizzazione del calcolo del pagerank
pthread_mutex_t mutex;
pthread_cond_t cond;
int counter = 0; // Contatore dei thread che hanno raggiunto il punto di sincronizzazione
int phase = 0;

int main(int argc, char *argv[]) {
    // Valori di default
    int K = 3;
    int M = 100;
    double D = 0.9;
    double E = 1.0e-8;
    int T = 3;
    char *infile = NULL;

    // Parse delle opzioni della linea di comando
    int opt;
    while ((opt = getopt(argc, argv, "k:m:d:e:t:")) != -1) {
        switch (opt) {
            case 'k':
                K = atoi(optarg);
                break;
            case 'm':
                M = atoi(optarg);
                break;
            case 'd':
                D = atof(optarg);
                break;
            case 'e':
                E = atof(optarg);
                break;
            case 't':
                T = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s [-k K] [-m M] [-d D] [-e E] [-t T] infile\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (optind < argc) {
        infile = argv[optind];
    } else {
        fprintf(stderr, "Usage: %s [-k K] [-m M] [-d D] [-e E] [-t T] infile\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    clock_t start = clock();

    // Inizializzazione della coda di task
    TaskQueue task_queue;
    initTaskQueue(&task_queue, 1024);

    // Inizializzazione del grafo
    grafo *g = (grafo *)malloc(sizeof(grafo));

    // Inizializzazione del thread pool
    ThreadPool pool;
    threadPoolInit(&pool, T, &task_queue, g);

    // Inizializzazione dei dati passati ai thread
    ThreadData data;
    data.task_queue = &task_queue;
    data.g = g;
    data.filename = (char *)infile;
    data.num_aux_threads = T;
    data.pool = &pool;

    // Creazione del thread produttore
    printf("Creating PRODUCER thread\n");
    pthread_t thread_producer;
    pthread_create(&thread_producer, NULL, produce, &data);

    // Creazione dei thread ausiliari consumatori
    for (int i = 0; i < pool.num_aux_threads; i++) {
        if (pthread_create(&pool.aux_threads[i], NULL, thread_function_graph, (void *)&pool) != 0) {
            perror("Failed to create thread");
            threadPoolDestroy(&pool);
            return EXIT_FAILURE;
        }
    }

    // Attesa della terminazione del thread produttore
    pthread_join(thread_producer, NULL);
    printf("PRODUCER finished\n");

    // Attesa della terminazione dei thread ausiliari
    threadPoolDestroy(&pool);
    destroyTaskQueue(&task_queue);

    printf("inizio il calcolo del pagerank\n");

    PageRankResult result = pagerank(g, g->N, D, E, M, T);
    printf("cacolo del pagerank per %d nodi completato\n", g->N);


    if(result.iterations == M) {
        printf("PageRank did not converge after %d iterations\n", M);
    } else {
        printf("Converged after %d iterations\n", result.iterations);
    }
    
    double sum_ranks = 0.0;
    for (int i = 0; i < g->N; ++i) {
        sum_ranks += result.ranks[i];
    }
    printf("Sum of ranks: %.4f (should be 1)\n", sum_ranks);

    // Stampa dei top K nodi per PageRank
    printf("Top %d nodes:\n", K);
    int *indices = (int *)malloc(g->N * sizeof(int));
    for (int i = 0; i < g->N; i++) {
        indices[i] = i;
    }

    // Ordinamento degli indici in base al PageRank
    qsort_r(indices, g->N, sizeof(int), compare, result.ranks);

    // Stampa i top K nodi
    for (int i = 0; i < K && i < g->N; ++i) {
        printf("  Node %d: rank = %.6f\n", indices[i], result.ranks[indices[i]]);
    }

    // Libera la memoria allocata
    free(indices);
    freeGraph(g);
    free(result.ranks);
    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tempo di esecuzione: %f secondi\n", elapsed_time);

    return 0;
}

// Inizializzazione delle variabili di sincronizzazione
void initialize_sync() {
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
}

// Distruzione delle variabili di sincronizzazione
void destroy_sync() {
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
}

// Funzione utilizzata per la sincronizzazione di thread durante il calcolo del pagerank
void barrier_wait(int num_threads) {
    pthread_mutex_lock(&mutex);
    counter++;
    if (counter == num_threads) {
        counter = 0;
        phase = 1 - phase;
        pthread_cond_broadcast(&cond);
    } else {
        int current_phase = phase;
        while (phase == current_phase) {
            pthread_cond_wait(&cond, &mutex);
        }
    }
    pthread_mutex_unlock(&mutex);
}

// Funzione che setta i parametri iniziali del pagerank e avvia il calcolo con i thread ausiliari
PageRankResult pagerank(grafo *g, int num_nodes, double damping, double epsilon, int maxiter, int num_threads) {
    printf("Calcolo del pagerank con %d nodi, damping %.2f, epsilon %.2e, maxiter %d, %d threads\n", num_nodes, damping, epsilon, maxiter, num_threads);
    double *ranks = (double *)malloc(num_nodes * sizeof(double));
    double *old_ranks = (double *)malloc(num_nodes * sizeof(double));
    double init_rank = 1.0 / num_nodes;
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    pthread_t signal__thread;

    for (int i = 0; i < num_nodes; i++) {
        ranks[i] = init_rank;
    }

    PagerankThreadData *thread_data = (PagerankThreadData *)malloc(num_threads * sizeof(PagerankThreadData));
    double *local_deltas = (double *)malloc(num_threads * sizeof(double));
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);
    // Maschera i segnali per tutti i thread eccetto il gestore dei segnali
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    pthread_sigmask(SIG_BLOCK, &set, NULL);

    int iterations = 0;
    int stop_flag = 0;

    // Crea il thread per la gestione dei segnali
    SignalThreadData signal_data = { &iterations, ranks, num_nodes, &lock };
    pthread_create(&signal__thread, NULL, signal_function, &signal_data);

    memcpy(old_ranks, ranks, num_nodes * sizeof(double));
    double *sum_dead_ends = (double *)malloc(sizeof(double));
    *sum_dead_ends = 0.0;
    for (int i = 0; i < num_nodes; i++) {
        if (g->out[i] == 0) {
            *sum_dead_ends += old_ranks[i];
        }
    }

    initialize_sync();

    double *delta = (double *)malloc(sizeof(double));
    *delta = 0.0;

    for (int i = 0; i < num_threads; i++) {
        // Calcolo dell'intervallo di nodi per ogni thread
        int start = i * (num_nodes / num_threads);
        int end = (i + 1) * (num_nodes / num_threads);
        if (i == num_threads - 1) {
            end = num_nodes;
        }
        // Inizializzazione dei dati passati ai thread
        thread_data[i].g = g;
        thread_data[i].ranks = ranks;
        thread_data[i].old_ranks = old_ranks;
        thread_data[i].damping = damping;
        thread_data[i].start = start;
        thread_data[i].end = end;
        thread_data[i].local_deltas = local_deltas;
        thread_data[i].thread_id = i;
        thread_data[i].iterations = &iterations;
        thread_data[i].epsilon = epsilon;
        thread_data[i].maxiter = maxiter;
        thread_data[i].stop_flag = &stop_flag;
        thread_data[i].num_threads = num_threads;
        thread_data[i].sum_dead_ends = sum_dead_ends; // Initial sum_dead_ends
        thread_data[i].delta = delta;
        pthread_create(&threads[i], NULL, pagerank_thread, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_cancel(signal__thread);
    pthread_join(signal__thread, NULL);

    // Normalizzazione dei ranks
    double total_rank = 0.0;
    for (int i = 0; i < num_nodes; i++) {
        total_rank += ranks[i];
    }
    for (int i = 0; i < num_nodes; i++) {
        ranks[i] /= total_rank;
    }

    PageRankResult result;
    result.ranks = ranks;
    result.iterations = iterations;
    result.delta = *delta; 

    destroy_sync();
    free(old_ranks);
    free(threads);
    free(thread_data);
    free(local_deltas);

    return result;
}

// Funzione eseguita dai thread per il calcolo del pagerank
void *pagerank_thread(void *arg) {
    PagerankThreadData *data = (PagerankThreadData *)arg;
    grafo *g = data->g;
    double *ranks = data->ranks;
    double *old_ranks = data->old_ranks;
    double damping = data->damping;
    int start = data->start;
    int end = data->end;
    double epsilon = data->epsilon;
    int maxiter = data->maxiter;
    double *local_deltas = data->local_deltas;
    int thread_id = data->thread_id;
    int *stop_flag = data->stop_flag;
    int *iterations = data->iterations;
    int num_threads = data->num_threads;
    double *sum_dead_ends = data->sum_dead_ends;
    double *delta = data->delta;


    while (1) {
    // Controllo del numero di iterazioni e del flag di stop
    if (*iterations >= maxiter || *stop_flag) {
        pthread_exit(NULL);
    }

    // Calcolo del pagerank per il sottoinsieme di nodi Fase1
    double local_delta = 0.0;
    for (int i = start; i < end; i++) {
        double rank_sum = 0.0;
        Node* current = g->in[i];
        while (current != NULL) {
            rank_sum += old_ranks[current->dest] / g->out[current->dest];
            current = current->next;
        }
        double new_rank = ((1.0 - damping) / g->N) + (damping * (rank_sum + *sum_dead_ends / g->N));
        ranks[i] = new_rank;
        local_delta += fabs(ranks[i] - old_ranks[i]);
    }
    local_deltas[thread_id] = local_delta;

    // Attendo che tutti i thread abbiano completato la fase 1
    barrier_wait(num_threads);

    // Aggiornamento variabili globali e aumento l'iterazione, esegiota solo da un thread Fase2
    if (thread_id == 0) {
        *delta = 0.0;
        for (int i = 0; i < num_threads; i++) {
            *delta += local_deltas[i];
        }
        if (*delta <= epsilon) {
            *stop_flag = 1;
        }

        (*iterations)++;
        memcpy(old_ranks, ranks, g->N * sizeof(double));
        *sum_dead_ends = 0.0;
        for (int i = 0; i < g->N; i++) {
            if (g->out[i] == 0) {
                *sum_dead_ends += old_ranks[i];
            }
        }
    }

    barrier_wait(num_threads);
    }

    return NULL;
}

// Funzione di confronto per l'ordinamento degli indici in base al PageRank
int compare(const void *a, const void *b, void *ranks) {
    int idx_a = *(int *)a;
    int idx_b = *(int *)b;
    double rank_a = ((double *)ranks)[idx_a];
    double rank_b = ((double *)ranks)[idx_b];
    return (rank_b > rank_a) - (rank_b < rank_a);
}

// Funzione eseguita dal thread per la gestione dei segnali
void *signal_function(void *arg) {
    SignalThreadData *data = (SignalThreadData *)arg;
    sigset_t set ;
    int sig;

    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);


    fprintf(stderr, "PID: %d \n",getpid());

    while (1) {
        int res = sigwait(&set, &sig);
        if (res != 0) {
            fprintf(stderr, "sigwait failed: %s\n", strerror(res));
            continue;
        }
        fprintf(stderr, "Signal received: %d\n", sig);

        if (sig == SIGUSR1) {
            pthread_mutex_lock(data->lock);

            int max_index = 0;
            double max_rank = data->ranks[0];
            // Trova il nodo con il PageRank maggiore
            for (int i = 1; i < data->num_nodes; i++) {
                if (data->ranks[i] > max_rank) {
                    max_rank = data->ranks[i];
                    max_index = i;
                }
            }

            fprintf(stderr, "Iterazione corrente: %d\n", *data->iterations);
            fprintf(stderr, "Nodo con il maggiore PageRank: %d\n", max_index);
            fprintf(stderr, "Valore del PageRank: %f\n", max_rank);

            pthread_mutex_unlock(data->lock);
        }
    }

    return NULL;
}

// Inizializzazione della coda di task
void initTaskQueue(TaskQueue *task_queue, int size) {
    task_queue->size = size;
    task_queue->in = 0;
    task_queue->out = 0;

    task_queue->buffer = mmap(NULL, size * sizeof(Arco), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (task_queue->buffer == MAP_FAILED) {
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }

    sem_init(&task_queue->empty, 1, size);
    sem_init(&task_queue->full, 1, 0);
    pthread_mutex_init(&task_queue->mutex, NULL);
    printf("Shared buffer initialized with size %d\n", size);
}

// Distruzione della coda di task
void destroyTaskQueue(TaskQueue *task_queue) {
    munmap(task_queue->buffer, task_queue->size * sizeof(Arco));
    sem_destroy(&task_queue->empty);
    sem_destroy(&task_queue->full);
    pthread_mutex_destroy(&task_queue->mutex);
}

// Inserimento di un arco nella coda di task, funzione eseguita solo dal producer
void enqueue(ThreadPool *pool, Arco arco) {
    TaskQueue *task = pool->task_queue;
    sem_wait(&task->empty);
    pthread_mutex_lock(&task->mutex);

    task->buffer[task->in] = arco;
    task->in = (task->in + 1) % task->size;

    pthread_mutex_unlock(&task->mutex);
    sem_post(&task->full);
}

// Estrazione di un arco dalla coda di task, funzione eseguita dai thread ausiliari consumer
Arco dequeue(ThreadPool *pool) {
    TaskQueue *task = pool->task_queue;
    sem_wait(&task->full);
    pthread_mutex_lock(&task->mutex);

    Arco arco = task->buffer[task->out];
    task->out = (task->out + 1) % task->size;

    pthread_mutex_unlock(&task->mutex);
    sem_post(&task->empty);
    return arco;
}

// Funzione eseguita dai thread ausiliari consumer
void *thread_function_graph(void *arg) {
    ThreadPool *pool = (ThreadPool *)arg;
    while (1) {
        Arco arco = dequeue(pool);
        if (arco.src == -1 && arco.dest == -1) {
            break; // Termina il thread
        }

        if (arco.src != arco.dest) {
            pthread_mutex_lock(&pool->g->mutex);
            addEdge(pool->g, arco.src, arco.dest);
            pthread_mutex_unlock(&pool->g->mutex);
        }
        usleep(1);
    }
    return NULL;
}

// Funzione eseguita dal thread produttore per la lettura del file e l'inserimento degli archi nella coda di task
void *produce(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    grafo *g = data->g;
    TaskQueue *task_queue = data->task_queue;

    FILE *file = fopen(data->filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Errore nell'apertura del file %s\n", data->filename);
        exit(1);
    }

    char line[1024];
    while (fgets(line, sizeof(line), file) != NULL && line[0] == '%');

    int r, c, n;
    sscanf(line, "%d %d %d", &r, &c, &n);
    if (r != c) {
        fprintf(stderr, "Errore: Matrice non quadrata\n");
        exit(1);
    }

    // Inizializzazione del grafo
    initGraph(g, r);
    int N = r;

    // Creazione dell'hashmap per evitare archi duplicati
    Hashmap *hashmap = createHashMap(N);

    // Lettura degli archi dal file e inserimento nella coda di task
    for (int i = 0; i < n; i++) {
        int src, dest;
        fscanf(file, "%d %d", &src, &dest);
        src--;
        dest--;

        if (src != dest && !hashmapContains(hashmap, src, dest)) {
            hashmapInsert(hashmap, src, dest);
            Arco arco = {src, dest};
            enqueue(data->pool, arco);
        } else {
            fprintf(stderr, "Arco illegale: (src=%d, dest=%d)\n", src + 1, dest + 1);
        }
    }

    fclose(file);
    printf("File reading completed\n");

    // Inserimento degli archi di terminazione nella coda di task
    for (int i = 0; i < data->num_aux_threads; i++) {
        Arco termination_arco = {-1, -1};
        enqueue(data->pool, termination_arco);
    }

    // Deallocazione della hashmap
    destroyHashmap(hashmap);

    return NULL;
}

//inizializzo il grafo
void initGraph(grafo *g, int N) {
    g->N = N;
    g->out = (int *)calloc(N, sizeof(int));
    g->in = (Node **)malloc(N * sizeof(Node *));
    for (int i = 0; i < N; i++) {
        g->in[i] = NULL;
    }
    printf("Graph initialized with %d nodes\n", N);
}

// Aggiunta di un arco al grafo
void addEdge(grafo *g, int src, int dest) {
    Node *newNode = (Node *)malloc(sizeof(Node));
    newNode->dest = src;
    newNode->next = g->in[dest];
    g->in[dest] = newNode;
    g->out[src]++;
}

// Deallocazione del grafo
void freeGraph(grafo *g) {
    for (int i = 0; i < g->N; i++) {
        Node *current = g->in[i];
        while (current != NULL) {
            Node *temp = current;
            current = current->next;
            free(temp);
        }
    }
    free(g->in);
    free(g->out);
    free(g);
}

// Inizializzazione del thread pool
void threadPoolInit(ThreadPool *pool, int num_aux_threads, TaskQueue *task_queue, grafo *g) {
    pool->num_aux_threads = num_aux_threads;
    pool->task_queue = task_queue;
    pool->g = g;
    pool->aux_threads = (pthread_t *)malloc(num_aux_threads * sizeof(pthread_t));
    pthread_mutex_init(&pool->g->mutex, NULL);
}

// Deallocazione del thread pool
void threadPoolDestroy(ThreadPool *pool) {
    for (int i = 0; i < pool->num_aux_threads; i++) {
        pthread_join(pool->aux_threads[i], NULL);
    }
    pthread_mutex_destroy(&pool->g->mutex);
    free(pool->aux_threads);
}

// Creazione dell'hashmap
Hashmap *createHashMap(int size) {
    Hashmap *set = (Hashmap *)malloc(sizeof(Hashmap));
    set->size = size;
    set->table = (HashmapNode **)malloc(size * sizeof(HashmapNode *));
    for (int i = 0; i < size; i++) {
        set->table[i] = NULL;
    }
    return set;
}

// Deallocazione dell'hashmap
void destroyHashmap(Hashmap *hashmap) {
    for (int i = 0; i < hashmap->size; i++) {
        HashmapNode *current = hashmap->table[i];
        while (current != NULL) {
            HashmapNode *temp = current;
            current = current->next;
            free(temp);
        }
    }
    free(hashmap->table);
    free(hashmap);
}

// Funzione di hash per l'hashmap
int hashFunction(int src, int dest, int size) {
    return (src * 31 + dest) % size;
}

// Verifica se l'hashmap contiene un arco
int hashmapContains(Hashmap *hashmap, int src, int dest) {
    int index = hashFunction(src, dest, hashmap->size);
    HashmapNode *current = hashmap->table[index];
    while (current != NULL) {
        if (current->src == src && current->dest == dest) {
            return 1;
        }
        current = current->next;
    }
    return 0;
}

// Inserimento di un arco nell'hashmap
void hashmapInsert(Hashmap *hashmap, int src, int dest) {
    int index = hashFunction(src, dest, hashmap->size);
    HashmapNode *new_node = (HashmapNode *)malloc(sizeof(HashmapNode));
    new_node->src = src;
    new_node->dest = dest;
    new_node->next = hashmap->table[index];
    hashmap->table[index] = new_node;
}
