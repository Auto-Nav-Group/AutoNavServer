from utils import json_to_obj
from nodegraph import NodeGraph
from pathfinder import PathFinder
import json

class Server:
    def __init__(self, hostip, port):
        self.hostip = hostip
        self.port = port
        self.httphandler = None
        self.map = None
        self.startnode = None
        self.endnode = None
        self.pathfinder = None
        self.nodegraph = None
        self.postcommands = {
            "update_map" : self.__local_update_map
        }
        self.getcommands = {
            "pathfind" : self.__local_pathfind
        }
    def start_server(self):
        from src.main.httpserverhandler import run_server
        self.httphandler = run_server(self.hostip, self.port)

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
            data = json.dumps(self.postcommands[command](post_data))
            return 200, data, 'application/json'
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
        return None

    def __local_pathfind(self):
        self.nodegraph = NodeGraph()
        self.nodegraph.create_from_map(self.map)
        self.pathfinder = PathFinder(self.nodegraph)
        return self.pathfinder.package_recalc_results(self.pathfinder.recalculate(self.nodegraph, self.startnode, self.endnode))