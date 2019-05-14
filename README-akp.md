# CarND-Project2-Advanced Lane Finding
In this project, i target to consolidate a code file "model-akp.py" that output result images of each state (such as camera calebration, Distortion correction,...) and finally write the annotated video. of the input video

I sincerely thank Jeremy-Shannon and Mohamedameen, for uploading their great work in Github. I downloaded their work on to my local machine and studied them very thoroughly. Understanding that, greatly helped me in completing my 2nd project in the Udacity nanodegree program


## Pipeline architecture:
- The total code has 3 major blocks
- 1st Block contain input (chessboard pictures and test pictures) and all the functions required for the total code to run such as
    - display, to define the display format
    - calibrate_camera
    - undistort
    - perspective transform
    - hls_l_thresh
    - lab_b_thresh
    - threshold_color_space
    - abs_sobel
    - mag_sobel
    - dir_sobel
    conbined_sobel
- 2nd Block is for defining the image processing pipeline. This includes various functions like
    - image_process
    - sliding_window
    - polyfit_prev_fit
    - curve_pos
    - draw_line
    - write_data
- 3rd and the last Block is for processing video. it includes main parts as follows
    - class_Line, for initialization
    - frame_processor, to link together all functions defined earlier
    - write annotated video from input video


### Result:
-  When the codes are run, all output images come first. Each image to be concelled for seeing the next image
-  Once all images are shown and cancelled, annotated video generation (of the input video) starts
-  For different input video files, the same is provided into the last block and outputannotated video file is found in the folder
-  total 3 videos are processed. they are visible in thye folder


#### Limitations and improvements

Annotation of input images worked without any problem
Easy videos also worked well
Difficult videos could not keep their lane identified continuoisly

## Conclusion

The major problem was to make a plan keeping functions seperate from the usages. Due to many functions, it waas very difficult to kep trtack of all functions
