from modules import op_colmap
import subprocess
import os

project_folder = r".\_project\test"
colmap_bin_path = r".\external\COLMAP\COLMAP-3.8-windows-cuda\COLMAP.bat"
colmap_camera_model = "SIMPLE_PINHOLE"
colmap_camera_params = ""



colmap_matcher = "sequential"
vocab_path = ""
aabb_scale = 32

# op_colmap.create_project_folders(project_folder)

# op_colmap.colmap_feature_extractor(colmap_bin_path, project_folder, colmap_camera_model, colmap_camera_params)

# op_colmap.colmap_match(colmap_bin_path, project_folder, colmap_matcher, vocab_path)

# op_colmap.colmap_mapper(colmap_bin_path, project_folder)

# op_colmap.colmap_bundle_adjuster(colmap_bin_path, project_folder)

# op_colmap.colmap_model_converter(colmap_bin_path, project_folder)

# camera_data = op_colmap.get_camera_data(project_folder)

# op_colmap.export_transforms_data(project_folder, aabb_scale, camera_data)


# subprocess.Popen('python C:\\_Dev\\Repository\\gaussian_splatting_webui\\repositories\\gaussian-splatting\\train.py -s C:\\_Dev\Repository\\gaussian_splatting_webui\\_project\\test', shell=True, stdout=subprocess.PIPE).stdout.read()



# os.system(python ".\repositories\gaussian-splatting\train.py" -s "C:\_Dev\Repository\gaussian_splatting_webui\_project\test")

# subprocess.run(["python", r"C:\_Dev\Repository\gaussian_splatting_webui\repositories\gaussian-splatting\train.py", "-s", r"C:\_Dev\Repository\gaussian_splatting_webui\_project\test"])

subprocess.Popen('python .\\repositories\\gaussian-splatting\\train.py -s C:\\_Dev\Repository\\gaussian_splatting_webui\\_project\\test', shell=True, stdout=subprocess.PIPE).stdout.read()
