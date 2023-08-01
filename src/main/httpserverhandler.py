from http.server import HTTPServer, ThreadingHTTPServer
from server import Server
import http.server


server = Server('localhost', 8000)

def get_server():
    return server

class JSONHandler(http.server.BaseHTTPRequestHandler):
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