import socket
import threading
import tempfile
import subprocess
import logging
import os
import signal

# Configurazione logging
logging.basicConfig(filename='server.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Impostazioni server
HOST = '127.0.0.1'
PORT = 51201
BUFFER_SIZE = 1024

# Funzione per gestire ogni connessione client
def handle_client(client_socket):
    try:
        # Ricezione del numero di nodi e archi
        n = int.from_bytes(client_socket.recv(4), byteorder='little')
        a = int.from_bytes(client_socket.recv(4), byteorder='little')
        print(f"Ricevuto: {n} nodi, {a} archi")
        
        archi_validi = 0
        archi_scartati = 0
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.mtx') as temp_file:
            print(f"File temporaneo: {temp_file.name}")
            temp_filename = temp_file.name
            #temp_file.write(f"%%MatrixMarket matrix coordinate real general\n".encode())
            temp_file.write(f"{n} {n} {a}\n")
            
            for _ in range(a):
                origin = int.from_bytes(client_socket.recv(4), byteorder='little')
                dest = int.from_bytes(client_socket.recv(4), byteorder='little')
                
                if 1 <= origin <= n and 1 <= dest <= n:
                    temp_file.write(f"{origin} {dest}\n")
                    archi_validi += 1
                else:
                    archi_scartati += 1
            
            temp_file.flush()        
        # Esecuzione di pagerank
        result = subprocess.run(['./pagerank', temp_filename], capture_output=True, text=True)
        
        # Risposta al client
        if result.returncode == 0:
            print (f"Pagerank calcolato con successo")
            client_socket.sendall(f"0 {result.stdout}".encode('utf-8'))
        else:
            print(f"Errore durante il calcolo di pagerank")
            client_socket.sendall(f"{result.returncode} {result.stderr}".encode('utf-8'))
        
        # Logging
        logging.info(f"Nodi: {n}, File temporaneo: {temp_filename}, Archi scartati: {archi_scartati}, Archi validi: {archi_validi}, Exit code: {result.returncode}")
    
    except Exception as e:
        logging.error(f"Errore nella gestione del client: {e}")
    
    finally:
        client_socket.close()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# Funzione per gestire il segnale SIGINT
def signal_handler(sig, frame):
    print("Bye dal server")
    server_socket.close()
    os._exit(0)

# Configurazione del segnale SIGINT
signal.signal(signal.SIGINT, signal_handler)

# Creazione del socket del server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print(f"Server in ascolto su {HOST}:{PORT}")

# Loop principale per accettare connessioni
while True:
    try:
        client_socket, addr = server_socket.accept()
        print(f"Connessione accettata da {addr}")
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()
    except KeyboardInterrupt:
        break

server_socket.close()
