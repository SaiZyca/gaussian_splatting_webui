import gradio as gr
from . import symbols, op_tinker, op_gaussian_splatting
import os

def gs_trainer_ui():
    with gr.Blocks() as ui:
        with gr.Accordion("Gaussian Splatting Trainer Settings", open=False):
            with gr.Group():
                gs_repo_path = gr.Textbox(label="Gaussian Splatting Repo path", value=r".\repositories\gaussian-splatting", interactive=True)
                with gr.Row(visible=False):
                    model_path = gr.Textbox(label='model_output_path')
                    resolution = gr.Dropdown(label='resolution', choices=['1', '1/2', '1/4', '1/8'], value='1', interactive=True)
                    convert = gr.Dropdown(label='convert', choices=['default', 'convert_cov3D_python', 'convert_SHs_python'], value='default', interactive=True)
                    sh_degree = gr.Number(label='sh_degree', value=3, minimum=1, maximum=3, interactive=True)
                with gr.Row(visible=False):
                    iterations = gr.Number(label='total iterations', value=30000)
                    save_iterations = gr.Textbox(label='save_iterations', value="7000 30000")
                    data_device = gr.Radio(label='data_device', choices=['GPU', 'CPU'], value='GPU', interactive=True)
                with gr.Row(visible=False):
                    start_checkpoint = gr.Textbox(label='start_checkpoint path')
                with gr.Row(visible=False):
                    checkpoint_iterations= gr.Checkbox(label='checkpoint_iterations')
                    white_background = gr.Checkbox(label="use white background instead of black", interactive=True)
                    eval = gr.Checkbox(label="use a MipNeRF360-style training", interactive=True)
        with gr.Group():
            with gr.Row():
                project_folder = gr.Textbox(label="Project Folder",  placeholder="path to COLMAP or NeRF Synthetic dataset", show_label=False, interactive=True)
                project_folder_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
            cmd_args = gr.Textbox(label='extra cmd args', value="")
            train_gs_btn = gr.Button(value="Process")
        
        train_gs_btn.click(op_gaussian_splatting.train_gaussian_splatting,
                                inputs=[gs_repo_path, project_folder, cmd_args], 
                                outputs=[], 
                                show_progress="hidden")
        
        project_folder_button.click(op_tinker.folder_browser,
                                inputs=[], 
                                outputs=project_folder, 
                                show_progress="hidden")

    return ui
    

def ui():
    with gr.Blocks() as ui:
        gs_trainer_ui()