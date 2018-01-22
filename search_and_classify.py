###############
# Load modules.
###############
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from helper_functions import *
from scipy.ndimage.measurements import label

#################################
# Read in car and non-car images.
#################################
t=time.time()
# Read in car images.
images = glob.glob('./vehicles/*/*.png')
#print('Numer of car images: ', len(images))
cars = []
for image in images:
    cars.append(image)
# Read in non-car images.
images = glob.glob('./non-vehicles/*/*.png')
#print('Number of non-car images: ', len(images))
notcars = []
for image in images:
    notcars.append(image)
#notcars = notcars[0:len(cars)]
#print('New number of non-car images: ', len(notcars))

# Plot the car/not-car example
#img_car = mpimg.imread(cars[0])
#img_notcar=mpimg.imread(notcars[0])
#fig = plt.figure()
#plt.subplot(121)
#plt.imshow(img_car, cmap='gray')
#plt.title('Example Car Image')
#plt.subplot(122)
#plt.imshow(img_notcar, cmap='gray')
#plt.title('Example Not-Car Image')
# Save the car/not-car example
#fig.savefig('./output_images/car_notcar.png')

# Reduce number of images for test runs
test_number = 500
cars = cars[0:test_number]
notcars = notcars[0:test_number]

t2 = time.time()
print(round(t2-t, 2), ' seconds to read in images.')

#############
# Parameters.
#############
# Choose parameters for feature extraction.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (8, 8) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
#y_start_stop = [400, 656] # Min and max in y to search in slide_window()
ystart = 400
ystop = 656
scale = 1.1

######################
# Features and labels.
######################
t=time.time()
# Extract features from images.
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

#Plot the HOG examples
#img_car = mpimg.imread(cars[0])
#img_notcar = mpimg.imread(notcars[0])
#gray_car = cv2.cvtColor(img_car, cv2.COLOR_RGB2GRAY)
#gray_notcar = cv2.cvtColor(img_notcar, cv2.COLOR_RGB2GRAY)
#features_car, hog_image_car = get_hog_features(gray_car, orient, pix_per_cell, cell_per_block, 
#                        vis=True, feature_vec=False)
#features_notcar, hog_image_notcar = get_hog_features(gray_notcar, orient, pix_per_cell, cell_per_block, 
#                        vis=True, feature_vec=False)
#plt.subplots_adjust(wspace=1,hspace=1)
#fig = plt.figure(figsize=(8,8))
#plt.subplot(221)
#plt.imshow(img_car, cmap='gray')
#plt.title('Example Car Image')
#plt.subplot(222)
#plt.imshow(hog_image_car, cmap='gray')
#plt.title('HOG Visualization Car')
#plt.subplot(223)
#plt.imshow(img_notcar, cmap='gray')
#plt.title('Example Not-Car Image')
#plt.subplot(224)
#plt.imshow(hog_image_notcar, cmap='gray')
#plt.title('HOG Visualization Not-Car')
# Save the HOG examples
#fig.savefig('./output_images/HOG_visualization.png')

t2 = time.time()
print(round(t2-t, 2), ' seconds to extract features.')

# Create an array stack of feature vectors.
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler.
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X.
scaled_X = X_scaler.transform(X)

# Define the labels vector.
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

#############
# Classifier.
#############
# Split up data into randomized training and test sets.
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
# Print data information.
print('Using spatial binning of:',spatial_size[0], ', ', hist_bins,'histogram bins, ', orient, 'orientations, ', pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block.')
print('Feature vector length:', len(X_train), '.')

# Use a SVC with GridSearch().
#parameters = {'C':[0.1, 1, 10]}
# Use a linear SVC. 
svc = LinearSVC(C=0.1)
#clf = GridSearchCV(svc, parameters)
# Check the training time for the SVC.
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'seconds to train SVC.')
#print(clf.best_params_)
# Check the score of the SVC.
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample.
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

#####################
#Search and classify.
#####################

image = mpimg.imread('./test_images/test6.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255

#windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
#                    xy_window=(64,64), xy_overlap=(0.5, 0.5))

#hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
#                        spatial_size=spatial_size, hist_bins=hist_bins, 
#                        orient=orient, pix_per_cell=pix_per_cell, 
#                        cell_per_block=cell_per_block, 
#                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                        hist_feat=hist_feat, hog_feat=hog_feat)                       

#window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

out_img, heat = find_cars(draw_image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)                

#plt.imsave('./output_images/test1_sliding_windows',out_img)

# Apply threshold to help remove false positives
heat = apply_threshold(heat,1)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img_2 = draw_labeled_bboxes(np.copy(image), labels)

# Plot the car Position together with the heatmap
#fig = plt.figure()
#plt.subplot(121)
#plt.imshow(draw_img_2)
#plt.title('Car Positions')
#plt.subplot(122)
#plt.imshow(heatmap, cmap='hot')
#plt.title('Heat Map')
# Save the car position together with the heatmap
#fig.savefig('./output_images/test6_car_position_heatmap.png')
