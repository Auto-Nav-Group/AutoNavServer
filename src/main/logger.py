import os
import platform
import socket
import time


class logger:
    logtypedefs = {
        "l": " | LOG | ",
        "e": " | ERROR | ",
        "w": " | WARNING | "
    }

    def __init__(self, name, path="/tmp/"):
        self.name = name
        self.path = path
        if self.path[len(self.path)-1] != '/':
            self.path+='/'
        try:
            os.remove(self.path + self.name)
        except:
            pass
        open(self.path + self.name, "x")
        initializationlogs = [
            'Started logging session',
            'System details: ' + platform.platform(),
            'Hostname: ' + socket.gethostname() + ', IP: ' + socket.gethostbyname(socket.gethostname())
        ]
        self.logs(initializationlogs)
    def log(self, msg, logtype="l"):
        '''
        Log type definitions:
        l = normal log
        e = error
        w = warning
        '''
        f = open(self.path + self.name, "a")
        for ldef in self.logtypedefs:
            if logtype == ldef:
                current_time = time.localtime()
                h = "{:2d}".format((current_time.tm_hour) % 24)
                m = "{:02d}".format(current_time.tm_min)
                s = "{:02d}".format(current_time.tm_sec)
                msg = f"{h}:{m}:{s}" + self.logtypedefs[ldef] + msg
        msg+='\n'
        f.write(msg)
        f.close()
    def logs(self, msgs, logtype="l"):
        '''
        Log type definitions:
        l = normal log
        e = error
        w = warning
        '''
        f = open(self.path + self.name, "a")
        writemsg = ''
        for msg in msgs:
            for ldef in self.logtypedefs:
                if logtype == ldef:
                    current_time = time.localtime()
                    h = "{:2d}".format((current_time.tm_hour - 4) % 24)
                    m = "{:02d}".format(current_time.tm_min)
                    s = "{:02d}".format(current_time.tm_sec)
                    writemsg += f"{h}:{m}:{s}" + self.logtypedefs[ldef] + msg
            writemsg+='\n'
        f.write(writemsg)
        f.close()

    def deleteLog(self):
        os.remove(self.path + self.name)
