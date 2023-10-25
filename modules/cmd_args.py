import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage",)
parser.add_argument("--lowram", action='store_true', help="load stable diffusion checkpoint weights to VRAM instead of RAM",)
parser.add_argument('--listen', type=str, default='127.0.0.1', help='IP to listen on for connections to Gradio',)
parser.add_argument('--username', type=str, default='', help='Username for authentication',)
parser.add_argument('--password', type=str, default='', help='Password for authentication',)
parser.add_argument('--server_port', type=int, default=44444, help='Port to run the server listener on',)
parser.add_argument('--inbrowser', action='store_true', help='Open in browser',)
parser.add_argument('--share', action='store_true', help='Share the gradio UI',)
parser.add_argument('--api', default=True,action='store_true', help='api mode',)