"""
Automatically set data paths for different servers
"""
import os
import socket
import pathlib

os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'

server_name = socket.gethostname()

if ('mit' in server_name) or (server_name in ['oreo', 'mars', 'twix', 'milo', 'ahoy']):
    paths = {
        "DATAPATH": ':'.join([
            '/data/ddmg/interseg/data', 
            '/storage/', 
            '/local/jjgo/data'
            ])
    }
else:
    paths = {}

for k,v in paths.items():
    os.environ[k] = str(v)