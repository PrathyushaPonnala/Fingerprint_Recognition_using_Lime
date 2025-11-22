# fine.py

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# ... (other functions and imports)
# Step 3: Load and preprocess fingerprint images
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception(f"Error loading image: {image_path}")
        # Add additional preprocessing steps if necessary
        return img
    except Exception as e:
        print(f"Error: {e}")
        return None

# Step 4: Feature extraction using HOG
def extract_features(img):
    try:
        features, _ = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
        return features.flatten()  # Flatten the features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None


# Step 5: Load and preprocess the dataset
def load_dataset():
    genuine_images = [
        'data/genuine/genuine1.jpg',
        'data/genuine/genuine2.jpg',
        'data/genuine/genuine3.jpg',
        'data/genuine/genuine4.jpg',
        'data/genuine/genuine5.jpg',
        'data/genuine/genuine6.jpg',
        'data/genuine/genuine7.jpg',
        'data/genuine/genuine8.jpg',
        'data/genuine/genuine9.jpg',
        'data/genuine/genuine10.jpg',
        'data/genuine/genuine11.jpg',
        'data/genuine/genuine12.jpg',
        'data/genuine/genuine13.jpg',
        'data/genuine/genuine14.jpg',
        'data/genuine/genuine15.jpg',
        'data/genuine/genuine16.jpg',
        'data/genuine/genuine17.jpg',
        'data/genuine/genuine18.jpg',
        'data/genuine/genuine19.jpg',
        'data/genuine/genuine20.jpg',
        'data/genuine/genuine21.jpg',
        'data/genuine/genuine22.jpg',
        'data/genuine/genuine23.jpg',
        'data/genuine/genuine24.jpg',
        'data/genuine/genuine25.jpg',
        'data/genuine/genuine26.jpg',
        'data/genuine/genuine27.jpg',
        'data/genuine/genuine28.jpg',
        'data/genuine/genuine29.jpg',
        'data/genuine/genuine30.jpg',
        'data/genuine/genuine31.jpg',
        'data/genuine/genuine32.jpg',
        'data/genuine/genuine33.jpg',
        'data/genuine/genuine34.jpg',
        'data/genuine/genuine35.jpg',
        'data/genuine/genuine36.jpg',
        'data/genuine/genuine37.jpg',
        'data/genuine/genuine38.jpg',
        'data/genuine/genuine39.jpg',
        'data/genuine/genuine40.jpg',
        'data/genuine/genuine41.jpg',
        'data/genuine/genuine42.jpg',
        'data/genuine/genuine43.jpg',
        'data/genuine/genuine44.jpg',
        'data/genuine/genuine45.jpg',
        'data/genuine/genuine46.jpg',
        'data/genuine/genuine47.jpg',
        'data/genuine/genuine48.jpg',
        'data/genuine/genuine49.jpg',
        'data/genuine/genuine50.jpg',
        'data/genuine/genuine51.jpg',
        'data/genuine/genuine52.jpg',
        'data/genuine/genuine53.jpg',
        'data/genuine/genuine54.jpg',
        'data/genuine/genuine55.jpg',
        'data/genuine/genuine56.jpg',
        'data/genuine/genuine57.jpg',
        'data/genuine/genuine58.jpg',
        'data/genuine/genuine59.jpg',
        'data/genuine/genuine60.jpg',
        'data/genuine/genuine61.jpg',
        'data/genuine/genuine62.jpg',
        'data/genuine/genuine63.jpg',
        'data/genuine/genuine64.jpg',
        # Add more genuine images as needed
    ]

    impostor_images = [
        'data/impostor/impostor1.jpg',
        'data/impostor/impostor2.jpg',
        'data/impostor/impostor3.jpg',
        'data/impostor/impostor4.jpg',
        'data/impostor/impostor5.jpg',
        'data/impostor/impostor6.jpg',
        'data/impostor/impostor7.jpg',
        'data/impostor/impostor8.jpg',
        'data/impostor/impostor9.jpg',
        'data/impostor/impostor10.jpg',
        'data/impostor/impostor11.jpg',
        'data/impostor/impostor12.jpg',
        'data/impostor/impostor13.jpg',
        'data/impostor/impostor14.jpg',
        'data/impostor/impostor15.jpg',
        'data/impostor/impostor16.jpg',
        'data/impostor/impostor17.jpg',
        'data/impostor/impostor18.jpg',
        'data/impostor/impostor19.jpg',
        'data/impostor/impostor20.jpg',
        'data/impostor/impostor21.jpg',
        'data/impostor/impostor22.jpg',
        'data/impostor/impostor23.jpg',
        'data/impostor/impostor24.jpg',
        'data/impostor/impostor25.jpg',
        'data/impostor/impostor26.jpg',
        'data/impostor/impostor27.jpg',
        'data/impostor/impostor28.jpg',
        'data/impostor/impostor29.jpg',
        'data/impostor/impostor30.jpg',
        'data/impostor/impostor31.jpg',
        'data/impostor/impostor32.jpg',
        'data/impostor/impostor33.jpg',
        'data/impostor/impostor34.jpg',
        'data/impostor/impostor35.jpg',
        'data/impostor/impostor36.jpg',
        'data/impostor/impostor37.jpg',
        'data/impostor/impostor38.jpg',
        'data/impostor/impostor39.jpg',
        'data/impostor/impostor40.jpg',
        'data/impostor/impostor41.jpg',
        'data/impostor/impostor42.jpg',
        'data/impostor/impostor43.jpg',
        'data/impostor/impostor44.jpg',
        'data/impostor/impostor45.jpg',
        'data/impostor/impostor46.jpg',
        'data/impostor/impostor47.jpg',
        'data/impostor/impostor48.jpg',
        'data/impostor/impostor49.jpg',
        'data/impostor/impostor50.jpg',
        'data/impostor/impostor51.jpg',
        'data/impostor/impostor52.jpg',
        'data/impostor/impostor53.jpg',
        # Add more impostor images as needed
    ]

    genuine_labels = np.ones(len(genuine_images))
    impostor_labels = np.zeros(len(impostor_images))

    images = genuine_images + impostor_images
    labels = np.concatenate([genuine_labels, impostor_labels])

    return images, labels

# Step 6: Train a machine learning model
def train_model(images, labels):
    features = [extract_features(preprocess_image(img)) for img in images if preprocess_image(img) is not None]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Step 7: Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    confusion_mat = confusion_matrix(y_test, predictions)

    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{confusion_mat}')

# ... (previous code)

# Step 8: Demo
def fingerprint_recognition_demo(model, test_image_path):
    # Load the test image
    test_image = preprocess_image(test_image_path)
    test_features = extract_features(test_image)

    # Original Prediction
    original_prediction = model.predict([test_features])[0]
    print("Original Prediction:")
    if original_prediction == 1:
        print("Genuine Fingerprint")
    else:
        print("Impostor Fingerprint")

    # LIME Explanation
    explainer = lime_image.LimeImageExplainer()

    # Ensure that the image is in the correct format for LIME
    test_image_for_lime = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)  # Move this line here

    # ... (previous code)

# Create a custom predictor function for LIME
def custom_predictor(images):
    # Reshape the images to match the expected input shape for the StandardScaler
    images_reshaped = images.reshape((images.shape[0], -1))

    # Use the StandardScaler to transform the reshaped images
    scaled_images = model.named_steps['standardscaler'].transform(images_reshaped)

    # Reshape the scaled images back to the original shape
    scaled_images_original_shape = scaled_images.reshape((scaled_images.shape[0], *images.shape[1:]))

    # Make predictions on the scaled images
    predictions = model.named_steps['svc'].predict(scaled_images_original_shape)

    return predictions

# ... (rest of the code)


if __name__ == "__main__":
    images, labels = load_dataset()

    model, X_test, y_test = train_model(images, labels)
    evaluate_model(model, X_test, y_test)

    # Test the model with a new fingerprint image
    test_image_path = 'data/test/test_fingerprint.jpg'
    fingerprint_recognition_demo(model, test_image_path)
