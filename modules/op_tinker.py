from tkinter import Tk, filedialog
import os

def file_browser(ext_filter:dict)->str:
    '''
    dict example:{'extension filter':[("video files","*.avi;*.mp4;*.mov;*.mkv"),]} \n
    return file path
    '''
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    result = ""

    filenames = filedialog.askopenfilenames(filetypes=ext_filter['extension filter'])
    if len(filenames) > 0:
        root.destroy()
        result = ", ".join(filenames)
        # return str(filenames)
    else:
        filename = "Files not seleceted"
        root.destroy()
        result = str(filename)
    
    return result

def folder_browser():
    '''
    return folder
    '''
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder = ""

    folder_path = filedialog.askdirectory()
    if folder_path:
        if os.path.isdir(folder_path):
            root.destroy()
            folder = str(folder_path)
        else:
            root.destroy()
            folder = str(folder_path)
    else:
        folder_path = "Folder not seleceted"
        root.destroy()
        folder = str(folder_path)
        
    return folder
        
def get_image_folder(exts:str):
    '''
    str example:{'extension filter':(".jpg",".png",".webp")} \n
    return file path
    '''
    folder = folder_browser()
    
    count = 0
    file_list = list()
    end_with = tuple(list(exts.split(",")))    
    
    for file in os.listdir(folder):
        if file.endswith(end_with):
            file_list.append(os.path.join(folder, file))
            count += 1
            
    return folder, count