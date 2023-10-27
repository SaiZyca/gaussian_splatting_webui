import os
import sys
import subprocess
import math
import json
import cv2
import numpy as np
import shutil
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPTS_FOLDER = os.path.join(ROOT_DIR, "scripts")

def train_gaussian_splatting(gs_repo_path, project_folder, cmd_args):
    trainer_path = r"%s\train.py" % gs_repo_path
    model_path = r"%s\gs" % project_folder
    
    command = "python %s --source_path %s --model_path %s %s" % (trainer_path, project_folder, model_path, cmd_args)
    
    try:
        result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        print ( "Error:\nreturn code: ", e.returncode, "\nOutput: ", e.stderr.decode("utf-8") )
        raise
    
    print ("========== train_gaussian_splatting Finished ==========")

def copy_files(source:list, destination_folder:str):
    file_list = list()
    for file in source:
        filename = os.path.basename(file)
        destination = r"%s\%s" % (destination_folder, filename) 
        result = shutil.copyfile(file, destination)
        file_list.append(result)
        
    return file_list

def colmap_feature_extractor(colmap_bin_path, project_folder, colmap_camera_model, colmap_camera_params):
    colmap_db = r"%s\colmap\colmap.db" % project_folder
    image_path = r"%s\images" % project_folder
    
    command = [colmap_bin_path, "feature_extractor",
                    "--ImageReader.camera_model", colmap_camera_model, 
                    "--ImageReader.camera_params", colmap_camera_params, 
                    "--SiftExtraction.estimate_affine_shape", "True",
                    "--SiftExtraction.domain_size_pooling", "True",
                    "--ImageReader.single_camera", "True",
                    "--database_path", colmap_db,
                    "--image_path", image_path,
                    ]
    
    # print (command)
    subprocess.run(command)
    
    print ("========== colmap_feature_extractor Finished ==========")

def colmap_match(colmap_bin_path, project_folder, colmap_matcher, vocab_path):
    colmap_db = r"%s\colmap\colmap.db" % project_folder
    
    command=[colmap_bin_path,
             "%s_matcher" % colmap_matcher,
             "--SiftMatching.guided_matching", "True",
             "--database_path", colmap_db,
            ]
    if vocab_path:
        command.append("--VocabTreeMatching.vocab_tree_path")
        command.append(vocab_path)
        
    # print (command)
    subprocess.run(command)
    
    print ("========== colmap_match Finished ==========")

def colmap_mapper(colmap_bin_path, project_folder, vocab_path=""):
    colmap_db = r"%s\colmap\colmap.db" % project_folder
    image_path = r"%s\images" % project_folder
    sparse_path = r"%s\sparse" % project_folder
    
    Path(sparse_path).mkdir(parents=True, exist_ok=True)
    
    command=[colmap_bin_path, "mapper",
             "--database_path", colmap_db,
             "--image_path", image_path,
             "--output_path", sparse_path
            ]
    if vocab_path:
        command.append("--VocabTreeMatching.vocab_tree_path")
        command.append(vocab_path)

    subprocess.run(command)
    
    print ("========== colmap_mapper Finished ==========")

def colmap_bundle_adjuster(colmap_bin_path, project_folder):
    sparse_path = r"%s\sparse\0" % project_folder
    
    command=[colmap_bin_path, "bundle_adjuster",
             "--input_path", sparse_path,
             "--output_path", sparse_path,
             "--BundleAdjustment.refine_principal_point", "1",
            ]

    subprocess.run(command)
    
    print ("========== colmap_bundle_adjuster Finished ==========")

def colmap_model_converter(colmap_bin_path, project_folder):
    sparse_path = r"%s\sparse\0" % project_folder
    output_path = r"%s\colmap\colmap_text" % project_folder
    
    command=[colmap_bin_path, "model_converter",
             "--input_path", sparse_path,
             "--output_path", output_path,
             "--output_type", "TXT",
            ]

    subprocess.run(command)
    
    print ("========== colmap_model_converter Finished ==========")

def run_colmap_project(project_folder, colmap_bin_path, colmap_matcher, colmap_camera_model, colmap_camera_params, vocab_path, aabb_scale, gs_repo_path, process_steps, cmd_args):
    if not Path(colmap_bin_path).is_file():
        return print("can't find colmap binary")
    if 'colmap' in process_steps:
        create_project_folders(project_folder)
        colmap_feature_extractor (colmap_bin_path, project_folder, colmap_camera_model, colmap_camera_params)
        colmap_match(colmap_bin_path, project_folder, colmap_matcher, vocab_path)
        colmap_mapper(colmap_bin_path, project_folder)
        colmap_bundle_adjuster(colmap_bin_path, project_folder)
        colmap_model_converter(colmap_bin_path, project_folder)
        camera_data = get_camera_data(project_folder)
        export_transforms_data(project_folder, aabb_scale, camera_data)
        
        print ("========== images colmap Finished ==========")
        
    if 'train gaussian splatting' in process_steps:
        train_gaussian_splatting(gs_repo_path, project_folder, cmd_args)
        
        print ("========== train gaussian splatting Finished ==========")
        
    print ("========== All Process Finished ==========")

def colmap_images(project_folder, files, \
    colmap_bin_path, colmap_matcher, colmap_camera_model, colmap_camera_params, vocab_path, aabb_scale):
    if not Path(colmap_bin_path).is_file():
        return print("can't find colmap binary")
    
    create_project_folders(project_folder)
    source = [file.name for file in files]
    image_path = r"%s\images" % (project_folder)
    copyed_files = copy_files(source, image_path)
    colmap_feature_extractor (colmap_bin_path, project_folder, colmap_camera_model, colmap_camera_params)
    colmap_match(colmap_bin_path, project_folder, colmap_matcher, vocab_path)
    camera_data = get_camera_data(project_folder)
    export_transforms_data(project_folder, aabb_scale, camera_data)
    
    # post_image_txt(project_folder, aabb_scale, SKIP_EARLY, OUT_PATH)

def create_project_folders(project_folder):
    
    Path(r"%s\images" % project_folder).mkdir(parents=True, exist_ok=True)
    Path(r"%s\colmap" % project_folder).mkdir(parents=True, exist_ok=True)
    Path(r"%s\colmap\colmap_text" % project_folder).mkdir(parents=True, exist_ok=True)
    Path(r"%s\sparse" % project_folder).mkdir(parents=True, exist_ok=True)
    
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def get_camera_data(project_folder):
    camera_txt = r"%s\colmap\colmap_text\cameras.txt" % project_folder   
    camera_data = dict()
    
    with open(camera_txt, "r") as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            k3 = 0
            k4 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            is_fisheye = False
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                is_fisheye = True
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                is_fisheye = True
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                is_fisheye = True
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                k3 = float(els[10])
                k4 = float(els[11])
            else:
                print("Unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi
            
        camera_data["w"] = w
        camera_data["h"] = h
        camera_data["fl_x"] = fl_x
        camera_data["fl_y"] = fl_y
        camera_data["k1"] = k1
        camera_data["k2"] = k2
        camera_data["k3"] = k3
        camera_data["k4"] = k4
        camera_data["p1"] = p1
        camera_data["p2"] = p2
        camera_data["cx"] = cx
        camera_data["cy"] = cy
        camera_data["is_fisheye"] = is_fisheye
        camera_data["angle_x"] = angle_x
        camera_data["angle_y"] = angle_y
        camera_data["fovx"] = fovx
        camera_data["fovy"] = fovy
        
    return camera_data
    # print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")
    
def export_transforms_data(project_folder, aabb_scale, camera_data:dict, SKIP_EARLY=0, keep_colmap_coords=True, mask_categories=[]):
    image_txt = r"%s\colmap\colmap_text\images.txt" % project_folder
    image_folder = r"%s\images" % project_folder
    OUT_PATH = r"%s\transforms.json" % project_folder
    
    with open(image_txt, "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_angle_x": camera_data["angle_x"],
            "camera_angle_y": camera_data["angle_y"],
            "fl_x": camera_data["fl_x"],
            "fl_y": camera_data["fl_y"],
            "k1": camera_data["k1"],
            "k2": camera_data["k2"],
            "k3": camera_data["k3"],
            "k4": camera_data["k4"],
            "p1": camera_data["p1"],
            "p2": camera_data["p2"],
            "is_fisheye": camera_data["is_fisheye"],
            "cx": camera_data["cx"],
            "cy": camera_data["cy"],
            "w": camera_data["w"],
            "h": camera_data["h"],
            "aabb_scale": aabb_scale,
            "frames": [],
        }

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIP_EARLY*2:
                continue
            if  i % 2 == 1:
                elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                #name = str(PurePosixPath(Path(image_folder, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(image_folder)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                b = sharpness(name)
                # print(name, "sharpness=",b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                if not keep_colmap_coords:
                    c2w[0:3,2] *= -1 # flip the y and z axis
                    c2w[0:3,1] *= -1
                    c2w = c2w[[1,0,2,3],:]
                    c2w[2,:] *= -1 # flip whole world upside down

                    up += c2w[0:3,1]
                
                name = os.path.relpath(name, start=os.path.dirname(OUT_PATH))
                frame = {"file_path":name,"sharpness":b,"transform_matrix": c2w}
                out["frames"].append(frame)
    nframes = len(out["frames"])

    if keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
    else:
        # don't keep colmap coords - reorient the scene to be easier to work with

        up = up / np.linalg.norm(up)
        # print("up vector was", up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # find a central point they are all looking at
        # print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3,:]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        # print(totp) # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        # print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes,"frames")
    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)

    if len(mask_categories) > 0:
        # Check if detectron2 is installed. If not, install it.
        try:
            import detectron2
        except ModuleNotFoundError:
            try:
                import torch
            except ModuleNotFoundError:
                print("PyTorch is not installed. For automatic masking, install PyTorch from https://pytorch.org/")
                sys.exit(1)

            input("Detectron2 is not installed. Press enter to install it.")
            import subprocess
            package = 'git+https://github.com/facebookresearch/detectron2.git'
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            import detectron2

        import torch
        from pathlib import Path
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        category2id = json.load(open(SCRIPTS_FOLDER / "category2id.json", "r"))
        mask_ids = [category2id[c] for c in args.mask_categories]

        cfg = get_cfg()
        # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo.
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        for frame in out['frames']:
            img = cv2.imread(frame['file_path'])
            outputs = predictor(img)

            output_mask = np.zeros((img.shape[0], img.shape[1]))
            for i in range(len(outputs['instances'])):
                if outputs['instances'][i].pred_classes.cpu().numpy()[0] in mask_ids:
                    pred_mask = outputs['instances'][i].pred_masks.cpu().numpy()[0]
                    output_mask = np.logical_or(output_mask, pred_mask)

            rgb_path = Path(frame['file_path'])
            mask_name = str(rgb_path.parents[0] / Path('dynamic_mask_' + rgb_path.name.replace('.jpg', '.png')))
            cv2.imwrite(mask_name, (output_mask*255).astype(np.uint8))
