# Yolo-Anipose Code
This code is meant to allow users to analyze videos with Yolov8 pose estimation and triangulate the results with Aniposelib.

## Improvements to be made
This code will not work if any frame in the video contains 0 people. It will only triangulate up until the frame when 0 people are present.

## Variables
`do_yolo` : If `True`, Yolov8 pose estimation will run on the videos listed in `vids`.\
`do_calibrate` : If `True`, the videos listed in `calib_vids` will be used to calibrate the cameras for triangulation. An output `.toml` file will save the
calibration results and the file name can be changed using `calibration_file`. This file is generated to prevent needing to recalibrate cameras before every
run.\
`do_triangulate` : If `True`, the `calibration_file` will be loaded to calibrate the cameras. Then, the data from `yolo_labels` will be read and triangulated.\
`projects` : Names of the projects that Yolov8 saves its results to.\
`calibration_file` : Name of a `.toml` file which stores camera calibration data.\
`calib_vids` : Paths to the videos used for camera calibration. The videos should start at the same time and be the same length (within a second) for best
results.\
`board` : Board object that is used to calibrate the cameras. See Aniposelib documentation for initializing different boards.\
`yolo_labels` :Paths to the `labels` folders that Yolov8 generates.\
`vids` : Names of the videos to be triangulated (do not include extension).\
`vid_path` : Paths to the videos in `vid`.\
`cam_names` : Names of the cameras.\
`pixel_width` : Width of the videos in `calib_vids` and `vids` measured in pixels.\
`pixel_height` : Height of the videos in `calib_vids` and `vids` measured in pixels.\
`fisheye_lens` : If `True`, the calibration will be done assuming the videos in `calib_vids` are taken with a fisheye lens.\
`n_bodyparts` : Number of bodyparts that are tracked (Yolov8 pose estimation tracks 17 bodyparts).\
`bodyparts` : List of bodypart names. This list must be in the order that is output from Yolov8 pose estimation.\
`plot_frame` : If `True`, plots one frame dictated by `framenum` and outputs to an interactive pyplot.\
`framenum` : Frame number to be plotted.\
`plot_3d` : If `True`, plots the 3D projection from an angled view and saves it to a video file at `output_file`.\
`plot_front` : If `True`, plots the 3D projection from the front and saves it to a video file at `output_file`.\
`plot_top` : If `True`, plots the 3D projection from the top and saves it to a video file at `output_file`.\
`plot_side` : If `True`, plots the 3D projection from the side and saves it to a video file at `output_file`.\
`output_file` : Path to where the plotted video file should be saved.\

## Things to Look Out For
1. `projects`, `calib_vids`, `yolo_labels`, `vids`, and `vid_path` should all have the same number of elements.
2. For checkerboards (in the `board` variable), the dimensions of the board are measured by the internal lattice points, not the number of squares. For
   example, a chessboard, which has 8 squares on each side, would be defined as `board = Checkerboard(7, 7, square_length=25)`. Additionally, the parameter
   `square_length` is in arbitrary units and only matters if the scale of the output matters.
3. For the best results, the calibration videos need to be good. The board should be moved through the entire field of view of all the cameras. The board
   should remain at a relatively constant distance from the cameras. The board should be held in the same orientation during the video.
