import threading

from utils import json_to_obj
from nodegraph import NodeGraph
from pathfinder import PathFinder
from threading import Thread, Timer
import asyncio
import json
import time
import sys

class Server:

    def __init__(self, hostip, port):
        self.hostip = hostip
        self.port = port
        self.httphandler = None
        self.server_running = False
        self.map = None
        self.startnode = None
        self.endnode = None
        self.pathfinder = None
        self.nodegraph = None
        self.httphandler = None
        self.shutdown_http = False
        self.shutdown_server = False
        self.handlerthread = None
        self.postcommands = {
            "/update_map" : self.__local_update_map,
            "/kill_thread" : self.__local_kill_thread
        }
        self.getcommands = {
            "/util/status" : self.__local_status,
            "/pathfind" : self.__local_pathfind,
            "/get_nodegraph" : self.__local_get_nodegraph
        }
    def start_server(self):
        try:
            from httpserverhandler import JSONHandler, ClosableHTTPServer
            self.httphandler = ClosableHTTPServer((self.hostip, self.port), lambda *args: JSONHandler(*args, proc_server=self), self)
            self.shutdown_http = False
            self.handlerthread = Thread(target=self.httphandler.serve_forever, name='HTTPHandler')
            self.handlerthread.daemon = True
            self.handlerthread.start()
        except Exception as e:
            print('Failed to run server. Exception ' + str(e))

    def handle_post(self, post_data, headers, path):
        try:
            command = path
        except Exception as e:
            print('Failed to process post request due to exception '+str(e))
            return 400, 'Command not specified', 'text/plain', None, None, None
        if command in self.postcommands:
            for header in headers:
                if header in vars(self):
                    vars(self)[header] = json_to_obj(headers[header])
            text, after_run_method, header2, header2content = self.postcommands[command](post_data)
            return 200, text, 'text/plain', after_run_method, header2, header2content
        else:
            return 400, 'Command Invalid', 'text/plain', None, None, None

    def handle_get(self, headers, path):
        try:
            command = path
        except Exception as e:
            print('Failed to process post request due to exception '+str(e))
            return 400, 'Command not specified', 'text/plain'
        if command in self.getcommands:
            for header in headers:
                if header in vars(self):
                    vars(self)[header] = json_to_obj(headers[header])
            data = json.dumps(self.getcommands[command](), indent=4)
            return 200, data, 'application/json'
        else:
            return 400, 'Command Invalid', 'text/plain'

    def __local_status(self):
        return {
            'server_running' : self.server_running,
        }

    def __local_update_map(self, data):
        self.map = json_to_obj(data)
        self.nodegraph = NodeGraph()
        self.nodegraph.create_from_map(self.map)
        return 'Updated map successfully', self.after_send_data, None, None

    def __local_pathfind(self):
        self.nodegraph = NodeGraph()
        self.nodegraph.create_from_map(self.map)
        self.pathfinder = PathFinder(self.nodegraph)
        return self.pathfinder.package_recalc_results(self.pathfinder.recalculate(self.nodegraph, self.startnode, self.endnode))

    def __local_kill_thread(self, post_data):
        return 'Killed thread successfully', self.__kill(), 'Connection', 'close'

    def __local_get_nodegraph(self):
        return self.nodegraph.to_json()

    def __kill(self):
        self.shutdown_server = True


    def after_send_data(self):
        pass

processingserver = Server('localhost', 8000)

if __name__ == "__main__":
    processingserver.start_server()
    while True:
        time.sleep(1)
        if processingserver.shutdown_server:
            break