\# UGV-DTU Software Departmental Test



This repository contains my code and work for the UGV-DTU Software Departmental Test.



\## Daily Learning Log



\### Day 1 25 august 2026



\- Learned how to create a GitHub repository.



\- Learned basic Git commands.



\- Learned how to add, commit and push files.



\### 26 August 2026



\#### What I did



\- Completed Task 1.



\- Created my GitHub repository.



\- Learned basic Git commands.



\- Created the required Task-1 folder structure.



\- Practiced push, pull, creating, renaming and deleting files.



\#### Problems I faced



\- Initially had trouble creating README.md in PowerShell.



\- Initially ran Git commands from the wrong folder.



\- Learned how to identify my current directory and initialize Git correctly.



\#### What I learned



\- How Git repositories work.



\- How to use git add, git commit and git push.



\- How to use basic PowerShell commands for files and folders.



\- How to organize the repository for future tasks.



\### 27 August 2026



\#### What I did



\- Completed Task 2: Lane Detection.

\- Implemented lane boundary detection using OpenCV.

\- Detected the left and right lane boundaries.

\- Highlighted the drivable area between the detected lane boundaries.

\- Added fallback methods for curved roads, low-visibility frames and partially occluded lanes.

\- Processed multiple input images automatically.

\- Saved the detected lane images separately in the Task-2/output folder.

\- Updated Task-2/main.py and verified that the program successfully processes the input images.



\#### Problems I faced



\- Initially, the program could not find the input images because of an incorrect folder/path setup.

\- Later, an error occurred while processing the images: `cannot unpack non-iterable numpy.int32 object`.

\- Debugged and corrected the code.

\- Verified that the final version processes the images successfully.



\#### What I learned



\- How lane detection can be implemented using OpenCV.

\- How Canny edge detection and Hough Line Transform can be used for lane detection.

\- How to define a Region of Interest for road images.

\- How to create a drivable-area overlay between lane boundaries.

\- How to automatically process and save multiple images using Python.

\- How to debug Python/OpenCV errors.



\#### Result



\- Task 2 completed successfully.

\- Input images were processed and the resulting lane-detected images were saved separately.



\### 27 August 2026



\#### Task 3 — Obstacle \& Pothole Detection



\##### What I did



\* Started Task 3 using Python and OpenCV.

\* Created the required `Task-3` folder structure with `Input` and `Output` folders.

\* Developed a computer vision program to detect potholes and obstacles in road images.

\* Implemented pothole detection using image preprocessing, thresholding, contours, and shape analysis.

\* Implemented obstacle detection using edge detection, morphological processing, contours, and filtering.

\* Added bounding boxes around detected potholes and obstacles.

\* Added object labels and pixel coordinates to the detected objects.

\* Added a summary showing the number of potholes and obstacles detected in each image.

\* Configured the program to automatically process all supported images from the `Input` folder.

\* Configured the program to save the processed images separately in the `Output` folder.

\* Tested and improved the detection after checking the problem images.

\* Successfully completed Task 3.



\##### Problems I faced



\* Initially, the Python code had formatting and indentation errors after copying the code.

\* Potholes were detected correctly, but obstacles were initially not being detected properly.

\* Tested the program on the problem images and adjusted the obstacle detection approach.

\* After modification and testing, obstacle detection worked correctly.



\##### What I learned



\* How contour-based object detection works in OpenCV.

\* How thresholding and edge detection can be used for computer vision tasks.

\* How bounding boxes can be used to mark detected objects.

\* How to calculate Intersection over Union (IoU) to identify overlapping detections.

\* How image preprocessing affects object detection.

\* How to process multiple images automatically using Python.

\* How to debug and improve a computer vision program by testing it on difficult images.



\##### Result



\* Task 3 completed successfully.

\* Potholes and obstacles are detected and marked in the output images.

\* Processed images are saved separately in the `Task-3/Output` folder.



