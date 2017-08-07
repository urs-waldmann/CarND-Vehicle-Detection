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

##########################
# Define helper functions.
##########################
# Define a function to compute binned color features.  
def bin_spatial(img, size):
    # Use cv2.resize().ravel() to create the feature vector.
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector.
    return features

# Define a function to compute color histogram features.  
def color_hist(img, nbins, bins_range):
    # Compute the histogram of the color channels separately.
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector.
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector.
    return hist_features

# Define a function to return HOG features and visualization.
##############################################################################
# ATTENTION: Set transform_sqrt=False if the image contains negative values!!!
##############################################################################
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis, feature_vec):
    # Call with two outputs if vis==True.
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output.
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
    return features

# Define a function to extract features from a list of images.
def extract_features(imgs, cspace, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel):
    # Create a list to append feature vectors to.
    features = []
    # Iterate through the list of images.
    for file in imgs:
        # Read in each one by one.
        image = mpimg.imread(file)
        # Apply color conversion if other than 'RGB'.
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)
        ###################################################################
        # OPTIONAL: Use also histograms of color and binned color features.
        ###################################################################
        ##################################################################################
        # Apply bin_spatial() to get spatial color features.
        #spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Append the new feature vector to the features list.
        #features.append(spatial_features)
        # Apply color_hist() also with a color space option now.
        #hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list.
        #features.append(hist_features)
        ##################################################################################
        # Call get_hog_feature.
        ##################################
        # Parameters: vis and feature_vec.
        ##################################
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list.
        features.append(hog_features)
    # Return list of feature vectors.
    return features

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
notcars = notcars[0:len(cars)]
#print('New number of non-car images: ', len(notcars))

# Reduce number of images for test runs
#test_number = 500
#cars = cars[0:test_number]
#notcars = notcars[0:test_number]

t2 = time.time()
print(round(t2-t, 2), ' seconds to read in images.')

#############
# Parameters.
#############
# Choose parameters for feature extraction.
spatial = 8
histbin = 6

colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12
pix_per_cell = 4
cell_per_block = 2
hog_channel = 1 # Can be 0, 1, 2, or "ALL"

######################
# Features and labels.
######################
t=time.time()
# Extract features from images.
car_features = extract_features(cars, cspace=colorspace, spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0, 256), orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
notcar_features = extract_features(notcars[0:len(cars)], cspace=colorspace, spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0, 256), orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

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
#print('Using spatial binning of:',spatial, ', ', histbin,'histogram bins, ', orient, 'orientations, ', pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block.')
print('Using', orient, 'orientations, ', pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block.')
print('Feature vector length:', len(X_train[0]), '.')

# Use a linear SVC. 
svc = LinearSVC()
# Check the training time for the SVC.
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC.
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample.
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

'''
# Use a SVC with GridSearch(). 
#parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10], 'gamma':[1, 100, 1000]}
parameters = {'kernel':['linear'], 'C':[0.1, 1, 10], 'gamma':[1, 100, 1000]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
# Check the training time for the SVC.
t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), ' seconds to train SVC.')
# Check the score of the SVC.
print('Test Accuracy of SVC = ', round(svr.score(X_test, y_test), 4))
# Check the prediction time for a single sample.
t=time.time()
n_predict = 10
print('My SVC predicts: ', svr.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), ' seconds to predict', n_predict,' labels with SVC')
'''
