from utils import json_to_obj
from nodegraph import NodeGraph
from pathfinder import PathFinder
from threading import Thread
import asyncio
import json
import time
import sys

class Server:
    def __init__(self, hostip, port):
        from http.server import HTTPServer
        from httpserverhandler import JSONHandler
        self.hostip = hostip
        self.port = port
        self.httphandler = None
        self.server_running = False
        self.map = None
        self.startnode = None
        self.endnode = None
        self.pathfinder = None
        self.nodegraph = None
        self.httphandler = HTTPServer((self.hostip, self.port), JSONHandler)
        self.postcommands = {
            "update_map" : self.__local_update_map,
            "kill_server" : self.__local_kill_server,
            "kill_thread" : self.__local_kill_thread
        }
        self.getcommands = {
            "pathfind" : self.__local_pathfind
        }
    def start_server(self):
        try:
            self.httphandler.serve_forever()
        except Exception as e:
            print('Failed to run server. Exception ' + str(e))

    def stop_server(self):
        try:
            print('Shutting down the server...')
            self.httphandler.shutdown()
        except Exception as e:
            print('Failed to stop server. Server may not be running. Exception ' + str(e))

    def handle_post(self, post_data, headers):
        try:
            command = headers['Command']
        except Exception as e:
            print('Failed to process post request due to exception '+str(e))
            return 400, 'Command not specified', 'text/plain'
        if command in self.postcommands:
            for header in headers:
                if header in vars(self):
                    vars(self)[header] = json_to_obj(headers[header])
            text, after_run_method = self.postcommands[command](post_data)
            return 200, text, 'text/plain', after_run_method
        else:
            return 400, 'Command Invalid', 'text/plain'

    def handle_get(self, headers):
        try:
            command = headers['Command']
        except Exception as e:
            print('Failed to process post request due to exception '+str(e))
            return 400, 'Command not specified', 'text/plain'
        if command in self.getcommands:
            for header in headers:
                if header in vars(self):
                    vars(self)[header] = json_to_obj(headers[header])
            data = json.dumps(self.getcommands[command]())
            return 200, data, 'application/json'
        else:
            return 400, 'Command Invalid', 'text/plain'

    def __local_update_map(self, data):
        self.map = json_to_obj(data)
        self.nodegraph = NodeGraph()
        self.nodegraph.create_from_map(self.map)
        return 'Updated map successfully', self.after_send_data

    def __local_pathfind(self):
        self.nodegraph = NodeGraph()
        self.nodegraph.create_from_map(self.map)
        self.pathfinder = PathFinder(self.nodegraph)
        return self.pathfinder.package_recalc_results(self.pathfinder.recalculate(self.nodegraph, self.startnode, self.endnode))

    def __local_kill_server(self, post_data):
        return 'Killed server successfully', self.stop_server()

    def __local_kill_thread(self, post_data):
        return 'Killed thread successfully', self.__kill()

    def __kill(self):
        self.stop_server()
        time.sleep(1)
        sys.exit()

    def after_send_data(self):
        pass

server = Server('localhost', 8000)

if __name__ == "__main__":
    server.start_server()