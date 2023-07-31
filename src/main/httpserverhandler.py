from json import loads
from nodegraph import NodeGraph
from pathfinder import PathFinder
from http.server import HTTPServer, ThreadingHTTPServer
from utils import JSONtoOBJ
from map import Map
from server import Server
import json
import http.server
import socketserver
import os, sys


server = Server('localhost', 8000)

class JSONHandler(http.server.BaseHTTPRequestHandler):
    def local_pathfind(self):
        self.vars["nodegraph"].createFromMap(self.vars["map"])
        self.vars["pathfinder"] = PathFinder(self.vars["nodegraph"])
        path, distance, time = self.vars["pathfinder"].recalculate(self.vars["nodegraph"], self.vars["startnode"], self.vars["endnode"])
        return path, distance
    def update_map(self, json):
        self.vars["map"]=JSONtoOBJ(json)
        self.vars["nodegraph"] = NodeGraph()
        self.vars["nodegraph"].createFromMap(self.vars["map"])
        return
    def __init__(self, request, client_address, server):
        self.vars = {
            "map" : Map(None),
            "nodegraph" : NodeGraph(),
            "startnode" : NodeGraph.Node(None),
            "endnode" : NodeGraph.Node(None),
            "pathfinder" : PathFinder(NodeGraph())
        }
        self.commands = {
            "pathfind" : self.local_pathfind,
            "update_map" : self.update_map
        }
        http.server.BaseHTTPRequestHandler.__init__(self, request, client_address, server)
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
        code, data, contenttype = server.handle_post(post_data, self.headers)
        self.send_response(code)
        self.send_header('Content-type', contenttype)
        self.end_headers()
        self.wfile.write(data.encode('utf-8'))
    def do_GET(self):
        code, data, contenttype = server.handle_get(self.headers)
        self.send_response(code)
        self.send_header('Content-type', contenttype)
        self.end_headers()
        self.wfile.write(data.encode('utf-8'))

def run_server(address, port):
    try:
        with ThreadingHTTPServer((address, port), JSONHandler) as httpd:
            httpd.serve_forever()
        return httpd
    except Exception as e:
        print('Failed to run server. Exception '+str(e))
if __name__ == "__main__":
    server.start_server()