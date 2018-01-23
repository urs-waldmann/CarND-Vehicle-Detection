# **Vehicle Detection Project**
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

In the following discussion I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation. The link to my GitHub repository can be found [here](https://github.com/urs-waldmann/CarND-Vehicle-Detection). 

---

In general I used to different versions of code. For my test runs on single images I used the files `search_and_classify.py` and `helper_functions.py` that I ran in my terminal and where I restricted the number of `vehicle` and `non-vehicle` images to 500. With this implementation I tested all my features on the single images provided and saved images for this writeup. Having a working code implementation I imported the needed pieces of code from those two files to the final Jupyter notebook `vehicle_detection.ipynb`. Here I extracted features from the whole car and not-car images provided and ran my implementaion on the `test_video.mp4` and later on my final `project_video_2.mp4` that I created in the "Advanced Lane Finding" project. For simplicity I will refer in this writeup to the two files `search_and_classify.py` and `helper_functions.py` and if necessary I will specify why some functions didn't make it into the final Jupyter notebook `vehicle_detection.ipynb`.

## Features and Labels

The code for this step is contained in lines 1 through 133 of the file called `search_and_classify.py`.

### Histogram of Oriented Gradients (HOG)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Example of a vehicle and non-vehicle](./output_images/car_notcar.png)

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orient`, `pix_per_cell`, and `cell_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.feature.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orient=9`, `pix_per_cell=(8, 8)` and `cell_per_block=(2, 2)`:

![Example of a HOG visualization](./output_images/HOG_visualization.png)

In addition to the histogram of orientated gradients I explored histograms of color and spatial binnings of color in order to use them as featurers for my classifier.

### Choice of parameters

I tried various combinations of parameters and chose the combination that gave me the best results training my classifier. My choice of parameters for the features mentioned above are the following:

| Parameter     | Value	        | 
|:-------------:|:-------------:| 
| Color space      | 'YCrCb'        | 
| HOG orientations      | 9        |
| HOG pixels per cell      | (8,8)      |
| HOG cells per block     | (2,2)      |
| HOG channel | "ALL" |
| Spatial binning dimensions | (8,8) |
| Number of histogram bins | 64 |

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

## Classifier

The code for this step is contained in lines 138 through 164 of the file called `search_and_classify.py`.

I trained a linear SVM using `sklearn.svm.LinaerSVC()`. In order to get the best result for my classifier without exploring the best parameters manually I used `sklearn.model_selection.GridSearchCV()`. Since I am using a linear SVM I can only tune the parameter `C`. Using `sklearn.model_selection.GridSearchCV()` with
```python
parameters = {'C':[0.1, 1, 10]}
```
I obtained `C=0.1` being the best fit for my model.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

