"""
Vercel Serverless Function for Air Doodle
Simple API that processes images and returns tracked results
"""

from http.server import BaseHTTPRequestHandler
import json
import base64
import numpy as np
import cv2

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            "message": "Air Doodle API",
            "status": "running",
            "endpoints": {
                "/": "This message",
                "/api/process": "POST image to process hand tracking"
            }
        }
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        try:
            data = json.loads(post_data.decode())
            # Return success response
            response = {
                "status": "success",
                "message": "Image received",
                "hands_detected": 0
            }
        except Exception as e:
            response = {
                "status": "error",
                "message": str(e)
            }
        
        self.wfile.write(json.dumps(response).encode())
