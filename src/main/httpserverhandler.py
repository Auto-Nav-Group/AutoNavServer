from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class JSONHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server, proc_server):
        self.proc_server = proc_server
        super().__init__(request, client_address, server)
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        code, text, contenttype, method, h2, h2content = self.proc_server.handle_post(post_data, self.headers, self.path)
        self.send_response(code)
        self.send_header('Content-type', contenttype)
        if h2 is not None and h2content is not None:
            self.send_header(h2, h2content)
        self.end_headers()
        self.wfile.write(bytes(text, 'utf-8'))

    def do_GET(self):
        code, data, contenttype = self.proc_server.handle_get(self.headers, self.path)
        self.send_response(code)
        self.send_header('Content-type', contenttype)
        self.end_headers()
        self.wfile.write(data.encode('utf-8'))

class ClosableHTTPServer(HTTPServer):
    def __init__(self, address, handler, parent_server):
        super().__init__(address, handler)
        self.parent_server = parent_server

    def serve_forever(self):
        while True:
            if self.parent_server.shutdown_http is True:
                break
            self.handle_request()
            if self.parent_server.shutdown_http is True:
                break