import gradio as gr
from . import op_colmap, symbols, op_tinker
import os

def colmap_ui():
    with gr.Blocks() as ui:
        with gr.Accordion("Colmap Settings", open=False):
            with gr.Row():
                colmap_bin_path = gr.Textbox(label="colmap binary path", value=r".\external\COLMAP\COLMAP-3.8-windows-cuda\COLMAP.bat", interactive=True)
            with gr.Row():
                colmap_matcher = gr.Dropdown(label="colmap matcher", value="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], interactive=True)
                colmap_camera_model = gr.Dropdown(label="colmap camera model", value="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], interactive=True)
                aabb_scale = gr.Number(label="aabb scale", value="32", minimum=1, maximum=128, step=2, interactive=True)
                video_fps = gr.Number(label="video fps", value="2", minimum=1, maximum=300, step=1, interactive=True)
            with gr.Row():
                out = gr.Textbox(label="Output path", value="transforms.json", interactive=True)
                vocab_path = gr.Textbox(label="Vocabulary tree path", value="")
                colmap_camera_params = gr.Textbox(label="colmap camera params", value="", placeholder="Format: fx,fy,cx,cy,dis", interactive=False)
        with gr.Accordion("ffmpeg Settings", open=False):
            with gr.Row():
                ffmpeg_bin_path = gr.Textbox(label="ffmpeg binary path", value=r".\external\ffmpeg\ffmpeg-6.0-essentials_build\bin\ffmpeg.exe", interactive=True)
                extract_format = gr.Radio(value=".png", choices=[".jpg",".png",".webp"], label="extract format",)
        with gr.Row():
            project_folder = gr.Textbox(label="Gaussain Splattiing Project Folder")
            project_folder_button = gr.Button(symbols.folder_symbol, elem_id='open_folder_small')
        with gr.Tab('Image Files'):
            image_files = gr.File(label='images', file_types=["image"], file_count='multiple')
            build_from_images_btn = gr.Button(value="Process")
        with gr.Tab('Movie'):
            movie_files = gr.File(label='Movie', file_types=["video"], file_count='multiple')
            build_from_video_btn = gr.Button()
        
        project_folder_button.click(op_tinker.folder_browser, 
                                inputs=[], 
                                outputs=project_folder, 
                                show_progress="hidden")
        
        build_from_images_btn.click(op_colmap.colmap_images,
                                    inputs=[project_folder, image_files, 
                                            colmap_bin_path, colmap_matcher, colmap_camera_model, colmap_camera_params, vocab_path, aabb_scale],
                                    outputs=[],
                                    )      

    return ui
    

def ui():
    with gr.Blocks() as ui:
        colmap_ui()