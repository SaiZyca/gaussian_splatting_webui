import subprocess


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