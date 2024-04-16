# This code is meant to allow users to analyze videos with Yolov8 pose and triangulate with Aniposelib.
# This code will not work if any frame in the video contains 0 people. It will only triangulate up until
# the frame when 0 people are present.
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from ultralytics import YOLO
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from matplotlib.pyplot import get_cmap
from matplotlib.animation import FuncAnimation

time_total1 = time.time()
#-------------------Variables that the user should edit-------------------#
# Runs Yolov8 pose analysis on 'vids'
do_yolo = True
# Calibrates cameras using 'calib_vids'. Saves a 'calibration_file' as a .toml file
do_calibrate = False
# Uses calibrated cameras from 'calibration_file' to triangulate 'vids' using 'yolo_labels'
do_triangulate = True
# Project names to store Yolo output
projects = ['p1', 'p2']
# Name of calibration file to save or load (must be .toml file)
calibration_file = 'calibration_final'
# Location of videos used to calibrate cameras (best to include full path). It is important that all
# videos are approximately the same length and start at the same time (within a few frames)
calib_vids=[['C:/Users/andre/OneDrive/Desktop/calib1A.mp4'],
            ['C:/Users/andre/OneDrive/Desktop/calib1B.mp4']]
# Board to be detected during calibration. Uses Aniposelib to make a checkerboard object (see
# aniposelib documentation for further info or to make charucoboards). The checkerboard size is
# defined as the number of internal verticies, not the number of squares: length, height
board = Checkerboard(8, 6, square_length=25)
# Location of Yolo labels folder (best to include full path)
yolo_labels = ['C:/Users/andre/source/repos/yolo/p1/predict5/labels/',
               'C:/Users/andre/source/repos/yolo/p2/predict5/labels/']
# Videos to be triangulated (do not include extension in 'vids'). It is important that all videos
# start at the same time (within a few frames). The length of the triangulation will be the length of
# the shortest video
vids = ['demo1A', 'demo1B']
vid_path = ['C:/Users/andre/OneDrive/Desktop/demo1A.mp4',
            'C:/Users/andre/OneDrive/Desktop/demo1B.mp4']
# Camera Names
cam_names = ['A', 'B']
# Camera Properties
pixel_width = 640
pixel_height = 480
fisheye_lens = False
# Bodypart settings (must match Yolo output order)
n_bodyparts = 17
bodyparts = ['nose', 'eyeL', 'eyeR', 'earL', 'earR', 'shL', 'shR', 'elbowL', 'elbowR',
             'wristL', 'wristR', 'hipL', 'hipR', 'kneeL', 'kneeR', 'ankleL', 'ankleR']
# Output settings (True to output, False to skip)
plot_frame = False
framenum = 100
plot_3d = True
plot_front = True
plot_top = True
plot_side = True
output_file = 'C:/Users/andre/OneDrive/Desktop/final_test.mp4'
#---------------End of variables that the user should edit----------------#

# Number of cameras
n_cams = len(calib_vids)

if do_yolo:
    time_yolo1 = time.time()
    model = YOLO('yolov8n-pose.pt')
    
    for i in range(n_cams):
        model.predict(vid_path[i], save=True, save_txt=True, project=projects[i])
        
    time_yolo2 = time.time()

# Calibrates cameras
if do_calibrate:
    time_calib1 = time.time()
    cgroup = CameraGroup.from_names(cam_names, fisheye=fisheye_lens)
    cgroup.calibrate_videos(calib_vids, board)
    # Saves calibration as 'calibration_file'
    cgroup.dump(calibration_file+'.toml')
    time_calib2 = time.time()

# Triangulating 'vids'
if do_triangulate:
    time_tri1 = time.time()
    # Loads 'calibration_file'
    cgroup = CameraGroup.load(calibration_file+'.toml')
    # Number of frames of all videos in 'vids'
    cam_frames = np.zeros(n_cams)
    # Yolo stores labels data as .txt files
    ext = '.txt'

    # Finds number of frames in each video
    for i in range(n_cams):
        dir_path = yolo_labels[i]
        count = 0
    
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
        
        # Prints file count for each video
        print('Cam', cam_names[i], 'frame count: ', count, '\n')
        cam_frames[i] = count

    # Use minimum number of frames so frames do not have to be fabricated for shorter videos
    frame_cnt = int(np.min(cam_frames))
    # 2D point data in the format of [cams[frames[bodyparts[point]]]]
    points = np.zeros((n_cams, frame_cnt, n_bodyparts, 2))
    # Confidence score data for each point in the format of [cams[frames[score of each bodypart]]]
    scores = np.zeros((n_cams, frame_cnt, n_bodyparts))

    for cam_num in range(n_cams):    
        filename = yolo_labels[cam_num]+vids[cam_num]+'_'
        i = 0
        
        while i < np.min(frame_cnt):
            f = open(filename+str(i+1)+ext, 'r')
            values = f.read().split()[5:]
        
            for j in range(n_bodyparts):
                points[cam_num][i][j][0] = pixel_width*float(values[3*j])
                points[cam_num][i][j][1] = pixel_height - pixel_height*float(values[3*j+1])
                scores[cam_num][i][j] = values[3*j+2]
        
            i += 1
    
    # Threshold for confidence scores
    score_threshold = 0.5
    # Redundant variables to double check sizes. n_cams should stay the same, n_points = frame_cnt,
    # n_joints = n_bodyparts
    n_cams, n_points, n_joints, _ = points.shape
    # Remove points that are below threshold
    points[scores < score_threshold] = np.nan
    # Code from Aniposelib Tutorial
    points_flat = points.reshape(n_cams, -1, 2)
    scores_flat = scores.reshape(n_cams, -1)
    # 'triangulate_ransac' is used because it is better at handling outliers, slower than 'triangulate'
    p3ds_flat = cgroup.triangulate_ransac(points_flat, progress=True)[0]
    reprojerr_flat = cgroup.reprojection_error(p3ds_flat, points_flat, mean=True)
    # p3ds contains 3D projected points
    p3ds = p3ds_flat.reshape(n_points, n_joints, 3)
    reprojerr = reprojerr_flat.reshape(n_points, n_joints)
    
    # Functions from Aniposelib Tutorial to connect bodyparts with lines
    def connect(ax, points, bps, bp_dict, color):
        ixs = [bp_dict[bp] for bp in bps]
        return ax.plot(points[ixs, 0], points[ixs, 1], points[ixs, 2], color=color)

    def connect_all(ax, points, scheme, bodyparts, cmap=None):
        if cmap is None:
            cmap = get_cmap('tab10')
        bp_dict = dict(zip(bodyparts, range(len(bodyparts))))
        lines = []
        for i, bps in enumerate(scheme):
            line = connect(ax, points, bps, bp_dict, color=cmap(i)[:3])
            lines.append(line)
        return lines

    # Connection scheme
    scheme = [
        ['eyeL', 'eyeR', 'nose', 'eyeL'],
        ['eyeL', 'earL', 'shL'],
        ['eyeR', 'earR', 'shR'],
        ['shL', 'shR', 'hipR', 'hipL', 'shL'],
        ['shL', 'elbowL', 'wristL'],
        ['shR', 'elbowR', 'wristR'],
        ['hipL', 'kneeL', 'ankleL'],
        ['hipR', 'kneeR', 'ankleR']]
    # Flip x values because Aniposelib mirrors the output
    p3ds[:, :, 0] = -p3ds[:, :, 0]
    # Mask coordinates to hide nan values
    x_masked = np.ma.masked_invalid(p3ds[:, :, 0])
    y_masked = np.ma.masked_invalid(p3ds[:, :, 1])
    z_masked = np.ma.masked_invalid(p3ds[:, :, 2])
    
    if plot_frame:
        # Plot one frame
        p3d = p3ds[frame_num]
        fig = plt.figure(figsize=(10, 12), constrained_layout=True)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        # Upper diagonal view
        ax1.set_xlim(np.min(x_masked), np.max(x_masked))
        ax1.set_ylim(np.min(y_masked), np.max(y_masked))
        ax1.set_zlim(np.min(z_masked), np.max(z_masked))
        ax1.view_init(elev=-45, azim=45, roll=45)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.scatter(x_masked[frame_num], y_masked[frame_num], z_masked[frame_num], c='black', s=100)
        connect_all(ax1, p3d, scheme, bodyparts)
        
        for x,y,z,a in zip(x_masked[frame_num], y_masked[frame_num], z_masked[frame_num], bodyparts):
            ax1.text(x, y, z, a)
            
        # Front view
        ax2.set_xlim(np.min(x_masked), np.max(x_masked))
        ax2.set_ylim(np.min(y_masked), np.max(y_masked))
        ax2.set_zlim(np.min(z_masked), np.max(z_masked))
        ax2.view_init(elev=-90, azim=90, roll=0)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        ax2.scatter(x_masked[frame_num], y_masked[frame_num], z_masked[frame_num], c='black', s=100)
        connect_all(ax2, p3d, scheme, bodyparts)
        
        for x,y,z,a in zip(x_masked[frame_num], y_masked[frame_num], z_masked[frame_num], bodyparts):
            ax2.text(x, y, z, a)
            
        # Top view
        ax3.set_xlim(np.min(x_masked), np.max(x_masked))
        ax3.set_ylim(np.min(y_masked), np.max(y_masked))
        ax3.set_zlim(np.min(z_masked), np.max(z_masked))
        ax3.view_init(elev=0, azim=90, roll=0)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')
        ax3.scatter(x_masked[frame_num], y_masked[frame_num], z_masked[frame_num], c='black', s=100)
        connect_all(ax3, p3d, scheme, bodyparts)
        
        for x,y,z,a in zip(x_masked[frame_num], y_masked[frame_num], z_masked[frame_num], bodyparts):
            ax3.text(x, y, z, a)
            
        # Side view
        ax4.set_xlim(np.min(x_masked), np.max(x_masked))
        ax4.set_ylim(np.min(y_masked), np.max(y_masked))
        ax4.set_zlim(np.min(z_masked), np.max(z_masked))
        ax4.view_init(elev=0, azim=0, roll=90)
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_zlabel('z')
        ax4.scatter(x_masked[frame_num], y_masked[frame_num], z_masked[frame_num], c='black', s=100)
        connect_all(ax4, p3d, scheme, bodyparts)
        
        for x,y,z,a in zip(x_masked[frame_num], y_masked[frame_num], z_masked[frame_num], bodyparts):
            ax4.text(x, y, z, a)
            
        plt.show()
    
    # Output all frames in a .mp4 file
    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Function used to animate output using FuncAnimation from matplotlib
    def animate(i):
        p3d = p3ds[i]

        # Upper diagonal view
        if plot_3d:
            ax1.clear()
            ax1.set_xlim(np.min(x_masked), np.max(x_masked))
            ax1.set_ylim(np.min(y_masked), np.max(y_masked))
            ax1.set_zlim(np.min(z_masked), np.max(z_masked))
            ax1.view_init(elev=-45, azim=45, roll=45)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.scatter(x_masked[i], y_masked[i], z_masked[i], c='black', s=100)
            connect_all(ax1, p3d, scheme, bodyparts)
            
            for x,y,z,a in zip(x_masked[i], y_masked[i], z_masked[i], bodyparts):
                ax1.text(x, y, z, a)

        # Front view
        if plot_front:
            ax2.clear()
            ax2.set_xlim(np.min(x_masked), np.max(x_masked))
            ax2.set_ylim(np.min(y_masked), np.max(y_masked))
            ax2.set_zlim(np.min(z_masked), np.max(z_masked))
            ax2.view_init(elev=-90, azim=90, roll=0)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')
            ax2.scatter(x_masked[i], y_masked[i], z_masked[i], c='black', s=100)
            connect_all(ax2, p3d, scheme, bodyparts)
            
            for x,y,z,a in zip(x_masked[i], y_masked[i], z_masked[i], bodyparts):
                ax2.text(x, y, z, a)

        # Top view
        if plot_top:
            ax3.clear()
            ax3.set_xlim(np.min(x_masked), np.max(x_masked))
            ax3.set_ylim(np.min(y_masked), np.max(y_masked))
            ax3.set_zlim(np.min(z_masked), np.max(z_masked))
            ax3.view_init(elev=0, azim=90, roll=0)
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_zlabel('z')
            ax3.scatter(x_masked[i], y_masked[i], z_masked[i], c='black', s=100)
            connect_all(ax3, p3d, scheme, bodyparts)
            
            for x,y,z,a in zip(x_masked[i], y_masked[i], z_masked[i], bodyparts):
                ax3.text(x, y, z, a)

        # Side view
        if plot_side:
            ax4.clear()
            ax4.set_xlim(np.min(x_masked), np.max(x_masked))
            ax4.set_ylim(np.min(y_masked), np.max(y_masked))
            ax4.set_zlim(np.min(z_masked), np.max(z_masked))
            ax4.view_init(elev=0, azim=0, roll=90)
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            ax4.set_zlabel('z')
            ax4.scatter(x_masked[i], y_masked[i], z_masked[i], c='black', s=100)
            connect_all(ax4, p3d, scheme, bodyparts)
            
            for x,y,z,a in zip(x_masked[i], y_masked[i], z_masked[i], bodyparts):
                ax4.text(x, y, z, a)
    
    # Animate frames
    ani = FuncAnimation(fig, animate, frames=frame_cnt, interval=1, repeat=False)
    # Save animation as .mp4
    ani.save(output_file, fps=15)
    time_tri2 = time.time()
    time_total2 = time.time()
    
    print('Total Time Taken (s): ', time_total2-time_total1, '\n')
    if do_yolo:
        print('Yolo Time Taken (s): ', time_yolo2-time_yolo1, '\n')
    if do_calibrate:
        print('Calib Time Taken (s): ', time_calib2-time_calib1, '\n')
    if do_triangulate:
        print('Triangulate Time Taken (s): ', time_tri2-time_tri1, '\n')