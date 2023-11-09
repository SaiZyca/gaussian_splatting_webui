#!/usr/bin/env python3
import gradio as gr
import os
from modules import cmd_args, ui_colmap, ui_gaussian_splatting
from pathlib import Path


# , ui_train_avatar
# def setup():
#     install_cmds = [
#         ['pip', 'install', 'gradio'],
#         ['pip', 'install', 'open_clip_torch'],
#         ['pip', 'install', 'clip-interrogator==0.6.0'],
#     ]
#     for cmd in install_cmds:
#         print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))

# setup()

def UI(**kwargs):
    css = ''

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file: 
            print('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(css=css, title='MoonShot Trainer GUI', theme=gr.themes.Default())

    root_path = Path().resolve()
    project_folder = r"%s\_project" % (root_path)
    os.makedirs(project_folder, exist_ok=True)

    with interface:
        with gr.Tab('Colmap to Gaussian Splatting'):
            ui_colmap.ui()

    # Show the interface
    launch_kwargs = {}
    username = kwargs.get('username')
    password = kwargs.get('password')
    server_port = kwargs.get('server_port', 0)
    inbrowser = kwargs.get('inbrowser', True)
    share = kwargs.get('share', False)
    server_name = kwargs.get('listen')

    launch_kwargs['server_name'] = server_name
    if username and password:
        launch_kwargs['auth'] = (username, password)
    if server_port > 0:
        launch_kwargs['server_port'] = server_port
    if inbrowser:
        launch_kwargs['inbrowser'] = inbrowser
    if share:
        launch_kwargs['share'] = share
        
    interface.launch(**launch_kwargs)

if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48) 
    args, _ = cmd_args.parser.parse_known_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        api=args.api,
        listen=args.listen,
    )
    
    