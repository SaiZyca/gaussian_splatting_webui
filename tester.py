from modules import op_colmap

project_path = r".\_project\test"
colmap_bin_path = r".\external\COLMAP\COLMAP-3.8-windows-cuda\COLMAP.bat"
colmap_camera_model = "SIMPLE_PINHOLE"
colmap_camera_params = ""
colmap_db = r"%s\colmap.db" % project_path
image_path = r"%s\images" % project_path


colmap_matcher = "sequential"
vocab_path = ""

# op_colmap.colmap_feature_extractor(colmap_bin_path, colmap_camera_model, colmap_camera_params, colmap_db, image_path)

# op_colmap.colmap_match(colmap_bin_path, colmap_matcher, colmap_db, vocab_path)

op_colmap.post_camera_txt(project_path)