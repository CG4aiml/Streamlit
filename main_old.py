import torch
import io
import os
import shutil
import pathlib
from PIL import Image
import cv2
import streamlit as st

content = ''
uploadedfilename = ''
extension = ''

#BEGIN MODEL SECTION
model = torch.hub.load('ultralytics/yolov5', 'custom', path='Checkpoints/bdd10Kv3.pt')  # local model

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.eval()
#END MODEL SECTION


#BEGIN UI SECTION
st.title('Object Detection and Tracking by IITH _:blue[Group4 2023]_')
  
with st.sidebar:
    with st.expander("Expand, Choose File (of type .jpg or mp4) and Upload", expanded=True):
        uploaded_file = st.file_uploader("Choose a file",['jpg','mp4'])
        if uploaded_file is not None:
            content = uploaded_file.getvalue()
            uploadedfilename = uploaded_file.name
            extension = pathlib.Path(uploadedfilename).suffix

        modelsel = st.radio("Model Selection", ('Yolov5', 'Yolov8'))
        tracksel = st.radio("Track Selection", ('Frame', 'GroupSort'))
        
        if(uploadedfilename != ''):
            if(extension == '.mp4'):
                st.header("Uploaded Video") 
                v = st.video(content)
            else:
                st.header("Uploaded Image") 
                orig = st.image(content)
        
        detectbtn = st.button("Detect/Track")
        metrics = st.checkbox("Show Metrics", key="disabled")
            
if detectbtn:
    if(uploadedfilename != ''):
        if(extension == '.mp4'):
            st.header("Processed Video")
            #TrackVideo Function call and then set the content
            detV = st.video(content)
        else:
            st.header("Processed Image")
            predict('Images(jpg)', content)        
            detI = st.image('image(jpg)', content)
#END UI SECTION

#BEGIN CODE SECTION
def predict(type, content):       
    if(os.path.exists('static/')):
        shutil.rmtree("static/")
         
    if(type == "video(mp4)"):
        createTempDirectories()
        FrameCapture(filename)
        generate_video()
    else:    
        detectImages(content, 'not required', 'not required', 'images')


def createTempDirectories():    
    os.makedirs("static/v")
    os.mkdir("static/d")
  
  
def detectImages(file, name, saveat, type):
    print("Image detection in progress")
    
    #img_bytes = file.read()
    #img = Image.open(io.BytesIO(img_bytes))
    results = model([file])

    results.render()  # updates results.imgs with boxes and labels
    results.save(save_dir=saveat)
    
    if(type == 'frames'):
        files = os.listdir(saveat)
    
        for f in files:
            os.rename(saveat + '/' + f, saveat + '/' + name)
            file = saveat + '/' + name
        
        shutil.move(file,'static/d')
        shutil.rmtree(saveat)
    
    print("Image detection complete")
  
  
def FrameCapture(path):
    print("Frame Capture in progress")
    print(path)
    # Path to video file
    vidObj = cv2.VideoCapture(path)
    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success = 1
  
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        # Saves the frames with frame-count
        cv2.imwrite("static/v/frame%d.jpg" % count, image)
        count += 1
    
    images = [img for img in os.listdir("static/v")
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
    
    for imgname in images:
        img = "static/v/" + imgname
        f = open(img, 'rb')
        detectImages(f, imgname, "static/temp", 'frames')
        
    print("Frame Capture complete")    
    
    
def generate_video():
    print("Video generation in progress")
    image_folder = 'static/d/' # make sure to use your folder
    video_name = 'static/generatedvideo.mp4'
      
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
     
    # Array images should only consider
    # the image files ignoring others if any
    print(images) 
  
    frame = cv2.imread(os.path.join(image_folder, images[0]))
  
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, fourcc, 100, (width, height)) 
  
    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated
    print("Video generation complete")

#END CODE SECTION   