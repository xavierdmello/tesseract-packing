"""Serve the website on saved results when the engine is not running:  python src/serve.py [port]"""
import os, sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
class H(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
    def end_headers(self):
        self.send_header("Cache-Control", "no-store"); super().end_headers()
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
print(f"http://localhost:{port}"); ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
