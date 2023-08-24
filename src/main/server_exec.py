from server import Server
import time

processing_server = Server('127.0.0.1', 8000)

processing_server.start_server()

while True:
    time.sleep(1)
    if processing_server.shutdown_server:
        break