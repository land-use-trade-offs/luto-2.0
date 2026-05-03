#!/usr/bin/env python3
"""TCP relay: forwards all connections from 0.0.0.0:PORT to 127.0.0.1:1080 (SOCKS5 proxy)"""
import socket, threading, sys

LISTEN_HOST = '0.0.0.0'
LISTEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 19080
TARGET_HOST = '127.0.0.1'
TARGET_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 1080

def relay(src, dst):
    try:
        while True:
            data = src.recv(4096)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try: src.close()
        except: pass
        try: dst.close()
        except: pass

def handle(client):
    try:
        server = socket.create_connection((TARGET_HOST, TARGET_PORT), timeout=10)
        t = threading.Thread(target=relay, args=(server, client), daemon=True)
        t.start()
        relay(client, server)
    except Exception as e:
        client.close()

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((LISTEN_HOST, LISTEN_PORT))
srv.listen(64)
print(f"SOCKS5 relay: {LISTEN_HOST}:{LISTEN_PORT} -> {TARGET_HOST}:{TARGET_PORT}", flush=True)
while True:
    client, addr = srv.accept()
    threading.Thread(target=handle, args=(client,), daemon=True).start()
