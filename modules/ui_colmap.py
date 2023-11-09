#!/usr/bin/env python3
import gradio as gr
from . import op_colmap, symbols, op_tinker, doc_cmd
import os



def colmap_ui():
    with gr.Blocks() as ui:
        with gr.Accordion("External Tool Settings", open=True):
            with gr.Tab("Colmap Settings"):
                with gr.Group():
                    with gr.Row():
                        colmap_bin_path = gr.Textbox(label="colmap binary path", value=r".\external\COLMAP\COLMAP-3.8-windows-cuda\COLMAP.bat", interactive=True)
                        colmap_cmd_args = gr.Textbox(label='colmap cmd args', value="", scale=2)
                    with gr.Row():
                        colmap_matcher = gr.Dropdown(label="colmap matcher", value="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], interactive=True)
                        colmap_camera_model = gr.Dropdown(label="colmap camera model", value="SIMPLE_PINHOLE", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], interactive=True)
                        aabb_scale = gr.Number(label="aabb scale", value="32", minimum=1, maximum=128, step=2, interactive=True)
                    with gr.Row(visible=False):
                        out = gr.Textbox(label="Output path", value="transforms.json", interactive=True, visible=True)
                        vocab_path = gr.Textbox(label="Vocabulary tree path", value="", visible=True)
                        colmap_camera_params = gr.Textbox(label="colmap camera params", value="", placeholder="Format: fx,fy,cx,cy,dis", interactive=False, visible=True)
            with gr.Tab("Gaussian Splatting Settings"):
                with gr.Group():
                    with gr.Row():
                        gs_repo_path = gr.Textbox(label="Gaussian Splatting Repo path", value=r".\repositories\gaussian-splatting", interactive=True)
                        gs_cmd_args = gr.Textbox(label='gs cmd args', value="", scale=2)
                    with gr.Accordion("Command doc", open=False):
                        gr.Markdown(value=doc_cmd.gs_cmd_doc)
            with gr.Tab("Agisoft Metashape Settings"):
                with gr.Row():
                    metashape_bin_path = gr.Textbox(label="Metashape binary path", value=r"C:\UserData\Appz\Agisoft\Metashape Pro 2.0.3\metashape.exe", interactive=True)
                    metashape_cmd_args = gr.Textbox(label='metashape cmd args', value="", scale=2)
                with gr.Row():
                    metashape_process_steps = gr.CheckboxGroup(label='Metashape Process steps',\
                        choices=['alignCameras','buildModel','buildTexture'], \
                        value=['alignCameras'])
        with gr.Group():
            with gr.Row():
                project_folder = gr.Textbox(label="Project Folder",  placeholder="Project Folder with /images", show_label=False)
                project_folder_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
            with gr.Row():
                process_steps = gr.CheckboxGroup(label='Process steps',choices=['colmap', 'metashape','train gaussian splatting'], value=['metashape'])
            process_btn = gr.Button(value="Process")
            process_log = gr.Textbox(value="log")
        
        project_folder_button.click(op_tinker.folder_browser,
                                inputs=[], 
                                outputs=project_folder, 
                                show_progress="hidden")
        
        process_btn.click(op_colmap.run_colmap_project,
                                    inputs=[project_folder, process_steps, \
                                            colmap_bin_path, colmap_matcher, colmap_camera_model, colmap_camera_params, vocab_path, aabb_scale, colmap_cmd_args, \
                                            gs_repo_path, gs_cmd_args, \
                                            metashape_bin_path, metashape_process_steps, metashape_cmd_args],
                                    outputs=[process_log],
                                    )      

    return ui
    

def ui():
    with gr.Blocks() as ui:
        colmap_ui()