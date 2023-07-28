from json import loads
from nodegraph import NodeGraph
from pathfinder import PathFinder
from http.server import HTTPServer, ThreadingHTTPServer
from utils import JSONtoOBJ
from map import Map
import json
import http.server
import socketserver
import os, sys

class JSONHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self):
        self.nodegraph = NodeGraph()
        self.startnode = None
        self.endnode = None
        self.map = None
        self.pathfinder = PathFinder(self.nodegraph)
        self.vars = {
            "map" : self.map,
            "nodegraph" : self.nodegraph,
            "startnode" : self.startnode,
            "endnode" : self.endnode
        }
        self.commands = {
            "pathfind" : self.pathfinder.recalculate(self.nodegraph, self.startnode, self.endnode),
        }
    def process_post_data(self, post_data):
        json_string = None
        json_data = None
        try:
            json_string = post_data.decode('utf-8')
            json_data = loads(json_string)
        except:
            print("Error decoding JSON data")
        return json_data

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        json_data = self.process_post_data(post_data)
        print("Received JSON data:")
        responsejson = None
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Received JSON data successfully")
    def do_GET(self):
        data = None
        try:
            command = self.headers['Command']
        except:
            command = None
            self.send_response(400, message=None)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Command header not found')
        if command in self.commands:
            for header in self.headers:
                if header in self.vars:
                    self.vars[header] = JSONtoOBJ(self.headers[header])
            data = json.dumps(self.commands[command](), default=lambda o: o.__dict__)
        self.send_response(200, message=None)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        if data is not None:
            self.wfile.write(data.encode('utf-8'))
        else:
            self.wfile.write(b'No data to send')

def run_server(address, port):
    try:
        with ThreadingHTTPServer((address, port), JSONHandler) as httpd:
            httpd.serve_forever()
        return httpd
    except Exception as e:
        print('Failed to run server. Exception '+str(e))
if __name__ == "__main__":
    run_server('localhost', 8000)