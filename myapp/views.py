from django.shortcuts import render, redirect,  get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .forms import *
from .models import *
from django.db.models import Q

from django.http import JsonResponse
from django.conf import settings
import os

import joblib
import numpy as np

def base(request):
    return render(request, 'base.html')

def about(request):
    return render(request, 'about/about.html')

def register(request):
    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        if user_form.is_valid():
            #create a new registration object and avoid saving it yet
            new_user = user_form.save(commit=False)
            #reset the choosen password
            new_user.set_password(user_form.cleaned_data['password'])
            #save the new registration
            new_user.save()
            return render(request, 'registration/register_done.html',{'new_user':new_user})
    else:
        user_form = UserRegistrationForm()
    return render(request, 'registration/register.html',{'user_form':user_form})

def profile(request):
    return render(request, 'profile/profile.html')



@login_required
def edit_profile(request):
    if request.method == 'POST':
        user_form = EditProfileForm(request.POST, request.FILES, instance=request.user)
        if user_form.is_valid():
            user = user_form.save()
            profile = user.profile
            if 'profile_picture' in request.FILES:
                profile.profile_picture = request.FILES['profile_picture']
            profile.save()
            messages.success(request, 'Your profile was successfully updated!')
            return redirect('profile')
    else:
        user_form = EditProfileForm(instance=request.user)
    
    return render(request, 'profile/edit_profile.html', {'user_form': user_form})

@login_required
def delete_account(request):
    if request.method == 'POST':
        request.user.delete()
        messages.success(request, 'Your account was successfully deleted.')
        return redirect('base')  # Redirect to the homepage or another page after deletion

    return render(request, 'registration/delete_account.html')
# das
@login_required
def dashboard(request):
    users_count = User.objects.all().count()
    consumers = Consumer.objects.all().count

    context = {
        'users_count':users_count,
        'consumers':consumers,
    }
    return render(request, "dashboard/dashboard.html", context=context)


# Contact start
@login_required
def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Thank you for contacting us!")
            return redirect('dashboard')  # Redirect to the same page to show the modal
    else:
        form = ContactForm()

    return render(request, 'contact/contact_form.html', {'form': form})

# contact end

import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from .forms import ImageUploadForm
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import io
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile

# Define the path to the models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'myapp', 'models', 'retinal_heart_risk_model.h5')

# Load the trained model (no need for joblib, directly use load_model)
model = load_model(MODEL_PATH)


import io
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile

def predict(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']
            
            # Convert the uploaded image for prediction
            image_bytes = io.BytesIO(uploaded_image.read())
            image = Image.open(image_bytes)
            image = image.resize((224, 224))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Make predictions
            predictions = model.predict(image)
            
            # Example true values
            true_values = [45, 130, 80, 26.5, 6.2]
            mae = np.mean(np.abs(np.array(true_values) - predictions[0]))
            
            predicted_age = predictions[0][0]
            predicted_sbp = predictions[0][1]
            predicted_dbp = predictions[0][2]
            predicted_bmi = predictions[0][3]
            predicted_hba1c = predictions[0][4]

            # Risk assessment logic
            risk_thresholds = {
                'sbp': 140,
                'dbp': 90,
                'bmi': 25,
                'hba1c': 6.5
            }
            is_at_risk = (
                predicted_sbp >= risk_thresholds['sbp'] or
                predicted_dbp >= risk_thresholds['dbp'] or
                predicted_bmi >= risk_thresholds['bmi'] or
                predicted_hba1c >= risk_thresholds['hba1c']
            )
            risk_status = "At Risk" if is_at_risk else "Healthy"
            
            # Save the prediction and true values to the database
            prediction_result = PredictionResult.objects.create(
                age=predicted_age,
                sbp=predicted_sbp,
                dbp=predicted_dbp,
                bmi=predicted_bmi,
                hba1c=predicted_hba1c,
                true_age=true_values[0],
                true_sbp=true_values[1],
                true_dbp=true_values[2],
                true_bmi=true_values[3],
                true_hba1c=true_values[4],
                mae=mae,
                risk_status=risk_status
            )
            
            # Prepare result data to render in the template
            result = {
                'age': predicted_age,
                'sbp': predicted_sbp,
                'dbp': predicted_dbp,
                'bmi': predicted_bmi,
                'hba1c': predicted_hba1c,
                'risk_status': risk_status,
                'risk_color': 'red' if risk_status == 'At Risk' else 'green',
                'true_values': true_values,
                'predicted_values': predictions[0],
                'mae': mae
            }

            # Render the results in the 'results.html' template
            return render(request, 'predictor/results.html', {'result': result})
    else:
        form = ImageUploadForm()

    return render(request, 'predictor/predict.html', {'form': form})

from django.shortcuts import render
import os
from django.conf import settings


def dataset(request):
    # Path to the images folder in the static directory
    image_folder_path = os.path.join(settings.BASE_DIR, 'static', 'dataset')  # Ensure 'dataset' folder exists under static

    # Ensure the folder exists
    if not os.path.exists(image_folder_path):
        return render(request, 'error.html', {'message': 'Dataset folder not found.'})

    # Get all image files in the folder
    images = [f for f in os.listdir(image_folder_path) if f.endswith(('png', 'jpg', 'jpeg', 'gif'))]

    # Pass image paths to the template
    image_paths = [f'dataset/{image}' for image in images]  # 'dataset' is a folder inside 'static'

    return render(request, 'dataset/dataset.html', {'images': image_paths})

def prediction_history(request):
    # Fetch all saved predictions from the database, ordered by creation time
    predictions = PredictionResult.objects.all().order_by('-created_at')
    return render(request, 'predictor/history.html', {'predictions': predictions})

import cv2
import numpy as np
from skimage import feature
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
import os

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    filtered = cv2.GaussianBlur(equalized, (5, 5), 0)
    return filtered

# Extract Region of Interest (ROI) using K-means clustering
def segment_image(image):
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(pixel_values)
    segmented_image = labels.reshape(image.shape)
    roi = (segmented_image == 1).astype(np.uint8)
    return roi

# Feature extraction: Mean, standard deviation, entropy, etc.
def extract_features(image, roi):
    features = {}
    roi_image = cv2.bitwise_and(image, image, mask=roi)
    features['mean'] = np.mean(roi_image)
    features['std'] = np.std(roi_image)
    features['variance'] = np.var(roi_image)
    features['entropy'] = stats.entropy(roi_image.flatten())
    glcm = feature.greycomatrix(roi_image, distances=[5], angles=[0], symmetric=True, normed=True)
    features['contrast'] = feature.greycoprops(glcm, 'contrast')[0, 0]
    features['correlation'] = feature.greycoprops(glcm, 'correlation')[0, 0]
    features['energy'] = feature.greycoprops(glcm, 'energy')[0, 0]
    features['homogeneity'] = feature.greycoprops(glcm, 'homogeneity')[0, 0]
    return features

# Load dataset, preprocess, extract features, and classify
def classify_heart_disease(features):
    # Example pre-trained RandomForest model (for demo)
    model = RandomForestClassifier()
    # Dummy training for simplicity (use a real dataset)
    X_dummy = np.random.rand(10, 8)
    y_dummy = np.random.randint(0, 2, size=(10,))
    model.fit(X_dummy, y_dummy)

    # Classify and return result
    prediction = model.predict([list(features.values())])
    probabilities = model.predict_proba([list(features.values())])[0]
    return prediction[0], probabilities



import cv2
import numpy as np
from skimage import feature
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
import os
from django.core.files.storage import FileSystemStorage

# Save intermediate images to display later
def save_image(image, name, base_path):
    image_path = os.path.join(base_path, name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)  # Ensure the directory exists
    cv2.imwrite(image_path, image)
    return os.path.join(settings.MEDIA_URL, name)  # Return media URL path


# Preprocess the image
def preprocess_image(image_path, save_path):
    image = cv2.imread(image_path)
    
    # Step 1: Query image (original)
    query_image_path = save_image(image, "query_image.png", save_path)

    # Step 2: Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image_path = save_image(gray, "grayscale_image.png", save_path)
    
    # Step 3: Histogram equalization
    equalized = cv2.equalizeHist(gray)
    equalized_image_path = save_image(equalized, "equalized_image.png", save_path)

    # Step 4: Filtering (Gaussian Blur)
    filtered = cv2.GaussianBlur(equalized, (5, 5), 0)
    filtered_image_path = save_image(filtered, "filtered_image.png", save_path)

    return filtered, {
        'query_image': query_image_path,
        'grayscale_image': grayscale_image_path,
        'equalized_image': equalized_image_path,
        'filtered_image': filtered_image_path
    }

# Extract Region of Interest (ROI) using K-means clustering
def segment_image(image, save_path):
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(pixel_values)
    segmented_image = labels.reshape(image.shape)

    # Save segmentation cluster image
    segmentation_image_path = save_image(segmented_image * 255, "segmentation_image.png", save_path)

    # ROI extraction
    roi = (segmented_image == 1).astype(np.uint8)
    roi_image_path = save_image(roi * 255, "roi_image.png", save_path)

    return roi, {
        'segmentation_image': segmentation_image_path,
        'roi_image': roi_image_path
    }

# Feature extraction: Mean, standard deviation, entropy, etc.
def extract_features(image, roi):
    features = {}
    roi_image = cv2.bitwise_and(image, image, mask=roi)
    features['mean'] = np.mean(roi_image)
    features['std'] = np.std(roi_image)
    features['variance'] = np.var(roi_image)
    features['entropy'] = stats.entropy(roi_image.flatten())
    glcm = feature.greycomatrix(roi_image, distances=[5], angles=[0], symmetric=True, normed=True)
    features['contrast'] = feature.greycoprops(glcm, 'contrast')[0, 0]
    features['correlation'] = feature.greycoprops(glcm, 'correlation')[0, 0]
    features['energy'] = feature.greycoprops(glcm, 'energy')[0, 0]
    features['homogeneity'] = feature.greycoprops(glcm, 'homogeneity')[0, 0]
    return features

# Load dataset, preprocess, extract features, and classify
def classify_heart_disease(features):
    # Example pre-trained RandomForest model (for demo)
    model = RandomForestClassifier()
    # Dummy training for simplicity (use a real dataset)
    X_dummy = np.random.rand(10, 8)
    y_dummy = np.random.randint(0, 2, size=(10,))
    model.fit(X_dummy, y_dummy)

    # Classify and return result
    prediction = model.predict([list(features.values())])
    probabilities = model.predict_proba([list(features.values())])[0]
    return prediction[0], probabilities

def heart_disease_detection_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            image_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            image_path = fs.path(filename)
            save_path = settings.MEDIA_ROOT  # Where to save the intermediate images

            # Preprocess the image
            preprocessed_image, preprocess_images = preprocess_image(image_path, save_path)

            # Segment the image
            roi, segmentation_images = segment_image(preprocessed_image, save_path)

            # Extract features
            features = extract_features(preprocessed_image, roi)

            # Classify the image
            risk_level, chance_percent = classify_heart_disease(features)

            # Define risk levels
            risk_labels = ["Low Chance", "Medium Chance", "High Chance"]
            risk_description = risk_labels[risk_level]

            # Combine all images for display
            all_images = {**preprocess_images, **segmentation_images}

            # Cleanup: Delete the uploaded image after processing (optional)
            if os.path.exists(image_path):
                os.remove(image_path)

            # Display the results in the template
            return render(request, 'test/heart_disease_result.html', {
                'risk': risk_description,
                'chance_percent': chance_percent,
                'features': features,
                'images': all_images
            })
    else:
        form = ImageUploadForm()

    return render(request, 'test/heart_disease_upload.html', {'form': form})
