import gradio as gr
from . import op_colmap, symbols, op_tinker
import os

def colmap_ui():
    with gr.Blocks() as ui:
        with gr.Accordion("Colmap Settings", open=False):
            with gr.Group():
                with gr.Row():
                    colmap_bin_path = gr.Textbox(label="colmap binary path", value=r".\external\COLMAP\COLMAP-3.8-windows-cuda\COLMAP.bat", interactive=True)
                with gr.Row():
                    colmap_matcher = gr.Dropdown(label="colmap matcher", value="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], interactive=True)
                    colmap_camera_model = gr.Dropdown(label="colmap camera model", value="SIMPLE_PINHOLE", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], interactive=True)
                    aabb_scale = gr.Number(label="aabb scale", value="32", minimum=1, maximum=128, step=2, interactive=True)
                with gr.Row():
                    out = gr.Textbox(label="Output path", value="transforms.json", interactive=True, visible=False)
                    vocab_path = gr.Textbox(label="Vocabulary tree path", value="", visible=False)
                    colmap_camera_params = gr.Textbox(label="colmap camera params", value="", placeholder="Format: fx,fy,cx,cy,dis", interactive=False, visible=False)
        with gr.Group():
            with gr.Row():
                project_folder = gr.Textbox(label="Project Folder",  placeholder="Project Folder with /images", show_label=False)
                project_folder_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
            with gr.Row():
                process_steps = gr.CheckboxGroup(label='Process setp',choices=['colmap','train gaussian splatting'], value=['colmap'])
            run_colmap_project_btn = gr.Button(value="Process")
        
        project_folder_button.click(op_tinker.folder_browser,
                                inputs=[], 
                                outputs=project_folder, 
                                show_progress="hidden")
        
        run_colmap_project_btn.click(op_colmap.run_colmap_project,
                                    inputs=[project_folder,colmap_bin_path, colmap_matcher, colmap_camera_model, colmap_camera_params, vocab_path, aabb_scale, process_steps],
                                    outputs=[],
                                    )      

    return ui
    

def ui():
    with gr.Blocks() as ui:
        colmap_ui()