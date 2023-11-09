import Metashape
import os, sys, time, shutil, struct, math
from pathlib import Path

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

def find_files(folder, types):
    return [entry.path for entry in os.scandir(folder) if (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]

if len(sys.argv) < 3:
    print("Usage: general_workflow.py <project_folder> <metashape_process_steps>")
    raise Exception("Invalid script arguments")

project_folder = sys.argv[1]
process_steps = sys.argv[2]

def metashape_procsee(project_folder, process_steps=[]):
    # inital
    doc = Metashape.Document()
    doc.save("%s/project.psx" % project_folder)
    
    image_folder = "%s/images" % project_folder
    photos = find_files(image_folder, [".jpg", ".jpeg", ".tif", ".tiff"])
    chunk = doc.addChunk()
    chunk.addPhotos(photos)
    
    doc.save()

    # print(str(len(chunk.cameras)) + " images loaded")

    if 'alignCameras' in process_steps:
        chunk.matchPhotos(keypoint_limit = 40000, tiepoint_limit = 10000, generic_preselection = True, reference_preselection = True)
        chunk.alignCameras()
        doc.save()
        
        print ("========== metashape alignCameras Finished ==========")
        
    if 'buildModel' in process_steps:
        chunk.buildDepthMaps(downscale = 2, filter_mode = Metashape.MildFiltering)
        chunk.buildModel(source_data = Metashape.DepthMapsData)
        doc.save()
        
        print ("========== metashape buildModel Finished ==========")

        has_transform = chunk.transform.scale and chunk.transform.rotation and chunk.transform.translation

        if has_transform:
            chunk.buildPointCloud()
            doc.save()

            chunk.buildDem(source_data=Metashape.PointCloudData)
            doc.save()

            chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)
            doc.save()

    if 'buildTexture' in process_steps:
        chunk.buildUV(page_count = 2, texture_size = 4096)
        chunk.buildTexture(texture_size = 4096, ghosting_filter = True)
        doc.save()
        
        print ("========== metashape buildTexture Finished ==========")
    # export results
    if "export" in process_steps:
        output_folder = "%s/output" % project_folder
        chunk.exportReport(output_folder + '/report.pdf')

        if chunk.model:
            chunk.exportModel(output_folder + '/model.obj')

        if chunk.point_cloud:
            chunk.exportPointCloud(output_folder + '/point_cloud.las', source_data = Metashape.PointCloudData)

        if chunk.elevation:
            chunk.exportRaster(output_folder + '/dem.tif', source_data = Metashape.ElevationData)

        if chunk.orthomosaic:
            chunk.exportRaster(output_folder + '/orthomosaic.tif', source_data = Metashape.OrthomosaicData)

        print('Processing finished, results saved to ' + output_folder + '.')

    return chunk

f32 = lambda x: bytes(struct.pack("f", x))
d64 = lambda x: bytes(struct.pack("d", x))
u8  = lambda x: x.to_bytes(1, "little", signed=(x < 0))
u32 = lambda x: x.to_bytes(4, "little", signed=(x < 0))
u64 = lambda x: x.to_bytes(8, "little", signed=(x < 0))
bstr = lambda x: bytes((x + "\0"), "utf-8")

def matrix_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if (tr > 0):
        s = 2 * math.sqrt(tr + 1)
        return Metashape.Vector([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        return Metashape.Vector([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if (m[1, 1] > m[2, 2]):
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        return Metashape.Vector([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        return Metashape.Vector([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])


def get_camera_name(cam):
    name = cam.label
    ext = os.path.splitext(name)
    if (len(ext[1]) == 0):
        name = ext[0] + os.path.splitext(cam.photo.path)[1]
    return name

def compute_undistorted_calib(sensor, zero_cxy):
    border = 0 # in pixels, can be increased if black margins are on the undistorted images

    if sensor.type != Metashape.Sensor.Type.Frame:
        return Metashape.Calibration()

    calib_initial = sensor.calibration
    w = calib_initial.width
    h = calib_initial.height

    calib = Metashape.Calibration()
    calib.f = calib_initial.f
    calib.width = w
    calib.height = h

    left = -float("inf")
    right = float("inf")
    top = -float("inf")
    bottom = float("inf")

    for i in range(h):
        pt = calib.project(calib_initial.unproject(Metashape.Vector([0.5, i + 0.5])))
        left = max(left, pt.x)
        pt = calib.project(calib_initial.unproject(Metashape.Vector([w - 0.5, i + 0.5])))
        right = min(right, pt.x)
    for i in range(w):
        pt = calib.project(calib_initial.unproject(Metashape.Vector([i + 0.5, 0.5])))
        top = max(top, pt.y)
        pt = calib.project(calib_initial.unproject(Metashape.Vector([i + 0.5, h - 0.5])))
        bottom = min(bottom, pt.y)

    left = math.ceil(left) + border
    right = math.floor(right) - border
    top = math.ceil(top) + border
    bottom = math.floor(bottom) - border

    if zero_cxy:
        new_w = min(2 * right - w, w - 2 * left)
        new_h = min(2 * bottom - h, h - 2 * top)
        new_w -= (new_w + w) % 2
        new_h -= (new_h + h) % 2
        left = (w - new_w) // 2
        right = (w + new_w) // 2
        top = (h - new_h) // 2
        bottom = (h + new_h) // 2

    calib.width = max(0, right - left)
    calib.height = max(0, bottom - top)
    calib.cx = -0.5 * (right + left - w)
    calib.cy = -0.5 * (top + bottom - h)

    return calib

def check_undistorted_calib(sensor, calib):
    border = 0 # in pixels, can be increased if black margins are on the undistorted images

    calib_initial = sensor.calibration
    w = calib.width
    h = calib.height

    left = float("inf")
    right = -float("inf")
    top = float("inf")
    bottom = -float("inf")

    for i in range(h):
        pt = calib_initial.project(calib.unproject(Metashape.Vector([0.5, i + 0.5])))
        left = min(left, pt.x)
        pt = calib_initial.project(calib.unproject(Metashape.Vector([w - 0.5, i + 0.5])))
        right = max(right, pt.x)
    for i in range(w):
        pt = calib_initial.project(calib.unproject(Metashape.Vector([i + 0.5, 0.5])))
        top = min(top, pt.y)
        pt = calib_initial.project(calib.unproject(Metashape.Vector([i + 0.5, h - 0.5])))
        bottom = max(bottom, pt.y)

    print(left, right, top, bottom)
    if (left < 0.5 or calib_initial.width - 0.5 < right or top < 0.5 or calib_initial.height - 0.5 < bottom):
        print("!!! Wrong undistorted calib")
    else:
        print("Ok:")

def get_coord_transform(frame, use_localframe):
    if not use_localframe:
        return frame.transform.matrix
    if not frame.region:
        print("Null region, using world crs instead of local")
        return frame.transform.matrix
    fr_to_gc  = frame.transform.matrix
    gc_to_loc = frame.crs.localframe(fr_to_gc.mulp(frame.region.center))
    fr_to_loc = gc_to_loc * fr_to_gc
    return (Metashape.Matrix.Translation(-fr_to_loc.mulp(frame.region.center)) * fr_to_loc)

def compute_undistorted_calibs(frame, zero_cxy):
    print("Calibrations:")
    calibs = {} # { sensor_key: ( sensor, undistorted calibration ) }
    for sensor in frame.sensors:
        calib = compute_undistorted_calib(sensor, zero_cxy)
        if (calib.width == 0 or calib.height == 0):
            continue
        calibs[sensor.key] = (sensor, calib)
        print(sensor.key, calib.f, calib.width, calib.height, calib.cx, calib.cy)
        #check_undistorted_calib(sensor, calib)

    return calibs

def get_calibs(camera, calibs):
    s_key = camera.sensor.key
    if s_key not in calibs:
        print("Camera " + camera.label + " (key = " + str(camera.key) + ") has cropped/unsupported sensor (key = " + str(s_key) + ")")
        return (None, None)
    return (calibs[s_key][0].calibration, calibs[s_key][1])


def get_filtered_track_structure(frame, folder, calibs):
    tie_points = frame.tie_points

    cnt_cropped = 0

    tracks = {} # { track_id: [ point indices, good projections, bad projections ] }; projection = ( camera_key, projection_idx )
    images = {} # { camera_key: [ camera, good projections, bad projections ] }; projection = ( undistored pt in pixels, size, track_id )
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        (calib0, calib1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue

        camera_entry = [cam, [], []]

        projections = tie_points.projections[cam]
        for (i, proj) in enumerate(projections):
            track_id = proj.track_id
            if track_id not in tracks:
                tracks[track_id] = [[], [], []]

            pt = calib1.project(calib0.unproject(proj.coord))
            good = (0 <= pt.x and pt.x < calib1.width and 0 <= pt.y and pt.y < calib1.height)
            place = (1 if good else 2)

            if not good:
                cnt_cropped += 1

            pos = len(camera_entry[place])
            camera_entry[place].append((pt, proj.size, track_id))
            tracks[track_id][place].append((cam.key, pos))

        images[cam.key] = camera_entry

    for (i, pt) in enumerate(tie_points.points):
        track_id = pt.track_id
        if track_id not in tracks:
            tracks[track_id] = [[], [], []]

        tracks[track_id][0].append(i)

    print("Found", cnt_cropped, "cropped projections")
    return (tracks, images)

def save_undistorted_images(params, frame, folder, calibs):
    folder = folder + "images/"
    T = Metashape.Matrix.Diag([1, 1, 1, 1])

    cnt = 0
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        if cam.sensor.key not in calibs:
            continue
        (calib0, calib1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue

        img = cam.image().warp(calib0, T, calib1, T)
        name = get_camera_name(cam)
        ext = os.path.splitext(name)[1]
        if ext.lower() in [".jpg", ".jpeg"]:
            c = Metashape.ImageCompression()
            c.jpeg_quality = params.image_quality
            img.save(folder + name, c)
        else:
            img.save(folder + name)
        cnt += 1
    print("Undistorted", cnt, "cameras")

def save_cameras(params, folder, calibs):
    export_file = r"%s/sparse/0/cameras.bin" % folder
    Path(export_file).parent.mkdir(parents=True, exist_ok=True)
    use_pinhole_model = params.use_pinhole_model
    with open(export_file, "wb") as fout:
        fout.write(u64(len(calibs)))
        for (s_key, (sensor, calib)) in calibs.items():
            fout.write(u32(s_key))
            fout.write(u32(1 if use_pinhole_model else 0))
            fout.write(u64(calib.width))
            fout.write(u64(calib.height))
            fout.write(d64(calib.f))
            if use_pinhole_model:
                fout.write(d64(calib.f))
            fout.write(d64(calib.cx + calib.width * 0.5))
            fout.write(d64(calib.cy + calib.height * 0.5))
    print("Saved", len(calibs), "calibrations")

# { camera_key: [ camera, good projections, bad projections ] }; projection = ( undistored pt in pixels, size, track_id )

def save_images(params, frame, folder, calibs, tracks, images):
    export_file = r"%s/sparse/0/images.bin" % folder
    Path(export_file).parent.mkdir(parents=True, exist_ok=True)
    only_good = params.only_good
    T_shift = get_coord_transform(frame, params.use_localframe)

    with open(export_file, "wb") as fout:
        fout.write(u64(len(images)))
        for (cam_key, [camera, good_prjs, bad_prjs]) in images.items():
            transform = T_shift * camera.transform
            R = transform.rotation().inv()
            T = -1 * (R * transform.translation())
            Q = matrix_to_quat(R)
            fout.write(u32(cam_key))
            fout.write(d64(Q.w))
            fout.write(d64(Q.x))
            fout.write(d64(Q.y))
            fout.write(d64(Q.z))
            fout.write(d64(T.x))
            fout.write(d64(T.y))
            fout.write(d64(T.z))
            fout.write(u32(camera.sensor.key))
            fout.write(bstr(get_camera_name(camera)))

            prjs = (good_prjs if only_good else good_prjs + bad_prjs)
            fout.write(u64(len(prjs)))
            for (pt, size, track_id) in prjs:
                track_id = (track_id if len(tracks[track_id][0]) != 1 else -1)
                fout.write(d64(pt.x))
                fout.write(d64(pt.y))
                fout.write(u64(track_id))
    print("Saved", len(images), "cameras")

# { track_id: [ point indices, good projections, bad projections ] }; projection = ( camera_key, projection_idx )

def save_points(params, frame, folder, calibs, tracks, images):
    export_file = r"%s/sparse/0/points3D.bin" % folder
    Path(export_file).parent.mkdir(parents=True, exist_ok=True)
    only_good = params.only_good
    T = get_coord_transform(frame, params.use_localframe)
    num_pts = len(list(filter(lambda x: len(x[0]) == 1, tracks.values())))

    with open(export_file, "wb") as fout:
        fout.write(u64(num_pts))
        for (track_id, [points, good_prjs, bad_prjs]) in tracks.items():
            if (len(points) != 1):
                continue
            point = frame.tie_points.points[points[0]]
            pt = T * point.coord
            track = frame.tie_points.tracks[track_id]
            fout.write(u64(track_id))
            fout.write(d64(pt.x))
            fout.write(d64(pt.y))
            fout.write(d64(pt.z))
            fout.write(u8(track.color[0]))
            fout.write(u8(track.color[1]))
            fout.write(u8(track.color[2]))
            fout.write(d64(0))

            num = (len(good_prjs) if only_good else len(good_prjs) + len(bad_prjs))
            fout.write(u64(num))
            for (camera_key, proj_idx) in good_prjs:
                fout.write(u32(camera_key))
                fout.write(u32(proj_idx))

            if not only_good:
                for (camera_key, proj_idx) in good_prjs:
                    fout.write(u32(camera_key))
                    fout.write(u32(proj_idx + len(images[camera_key][1])))
    print("Saved", num_pts, "points from", len(tracks), "tracks")


class ExportSceneParams():
    def __init__(self):
        # default values for parameters
        self.all_chunks = True
        self.all_frames = False

        self.zero_cxy = True
        self.use_localframe = True
        self.image_quality = 90
        self.confirm_deletion = True
        self.use_pinhole_model = True
        self.only_good = True

    def log(self):
        print("All chunks:", self.all_chunks)
        print("All frames:", self.all_frames)
        print("Zero cx and cy:", self.zero_cxy)
        print("Use local coordinate frame:", self.use_localframe)
        print("Image quality:", self.image_quality)
        print("Confirm deletion:", self.confirm_deletion)
        print("Using pinhole model instead of simple_pinhole:", self.use_pinhole_model)
        print("Using only uncropped projections:", self.only_good)


def export_for_gaussian_splatting(output_folder, chunk, params = ExportSceneParams()):
    print ("Start export_for_gaussian_splatting ")
    log_result = lambda x: print("", x, "-----------------------------------", sep="\n")
    params.log()

    frame_cnt = 0

    for frame_id, frame in enumerate(chunk.frames):
        if not frame.tie_points:
            continue
        if not params.all_frames and not (frame == chunk.frame):
            continue
        frame_cnt += 1

        calibs = compute_undistorted_calibs(frame, params.zero_cxy)
        (tracks, images) = get_filtered_track_structure(frame, output_folder, calibs)

        # save_undistorted_images(params, frame, output_folder, calibs)
        save_cameras(params, output_folder, calibs)
        save_images(params, frame, output_folder, calibs, tracks, images)
        save_points(params, frame, output_folder, calibs, tracks, images)

    log_result("export for gaussian splatting Finish")


label = "Scripts/Export Colmap project (for Gaussian Splatting)"
print("To execute this script press {}".format(label))

matched_chunk = metashape_procsee(project_folder, process_steps)
export_for_gaussian_splatting(project_folder, matched_chunk)