import gradio as gr
from . import op_colmap, symbols, op_tinker

def ffmpeg_ui():
    with gr.Blocks() as ui:
        with gr.Accordion("ffmpeg Settings", open=False):
            with gr.Row():
                ffmpeg_bin_path = gr.Textbox(label="ffmpeg binary path", value=r".\external\ffmpeg\ffmpeg-6.0-essentials_build\bin\ffmpeg.exe", interactive=True)
                extract_format = gr.Radio(value=".png", choices=[".jpg",".png",".webp"], label="extract format",)
            with gr.Row():
                with gr.Tab('Image Files'):
                    image_files = gr.File(label='images', file_types=["image"], file_count='multiple')
                    build_from_images_btn = gr.Button(value="Process")
                with gr.Tab('Movie'):
                    movie_files = gr.File(label='Movie', file_types=["video"], file_count='multiple')