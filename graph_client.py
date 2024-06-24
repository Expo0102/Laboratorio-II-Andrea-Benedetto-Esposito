import socket
import threading
import logging
import sys

# Configurazione logging
logging.basicConfig(filename='client.log', level=logging.INFO, format='%(asctime)s - %(message)s')

HOST = '127.0.0.1'
PORT = 51201
BUFFER_SIZE = 1024

def send_graph_data(filename):
    client_socket = None
    try:

        print(f"{filename} Inizio")
        with open(filename, 'r') as file:
            lines = file.readlines()
            #ometto i commenti
            while lines[0].startswith("%"):
                lines.pop(0)
        
        # Estrarre numero di nodi e archi
        header = lines[0].strip().split()
        n, a = int(header[0]), int(header[2])
        edges = [tuple(map(int, line.strip().split())) for line in lines[1:]]
        
        # Creare connessione al server
        print(f"{filename} Connessione a {HOST}:{PORT}")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        
        # Inviare numero di nodi e archi

        client_socket.sendall(n.to_bytes(4, byteorder='little'))
        client_socket.sendall(a.to_bytes(4, byteorder='little'))
        #client_socket.sendall(f"{n} {a}\n".encode('utf-8'))
        
        # Inviare archi
        
        for origin, dest in edges:
                client_socket.sendall(origin.to_bytes(4, byteorder='little'))
                client_socket.sendall(dest.to_bytes(4, byteorder='little'))

        # Ricevere risposta dal server
        response = client_socket.recv(BUFFER_SIZE).decode('utf-8')
        
        # Stampare la risposta con il prefisso del nome del file
        for line in response.strip().split('\n'):
            print(f"{filename} {line}")
        
        print(f"{filename} Bye")
        
    except Exception as e:
        logging.error(f"Errore nell'invio dei dati del grafo da {filename}: {e}")
    
    finally:
        if client_socket:
            client_socket.close()

def main():
    if len(sys.argv) < 2:
        print("Uso: python graph_client.py <file1.mtx> <file2.mtx> ...")
        sys.exit(1)
    
    files = sys.argv[1:]
    threads = []
    
    for filename in files:
        thread = threading.Thread(target=send_graph_data, args=(filename,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
