from http.server import HTTPServer, ThreadingHTTPServer
from threading import Thread
import http.server

class JSONHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        from server import server
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        code, text, contenttype, method = server.handle_post(post_data, self.headers)
        self.send_response(code)
        self.send_header('Content-type', contenttype)
        self.end_headers()
        self.wfile.write(bytes(text, 'utf-8'))
        if method is not None:
            method()

    def do_GET(self):
        from server import server
        code, data, contenttype = server.handle_get(self.headers)
        self.send_response(code)
        self.send_header('Content-type', contenttype)
        self.end_headers()
        self.wfile.write(data.encode('utf-8'))