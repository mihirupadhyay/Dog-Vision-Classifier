# Dog Breed Classification Project
# Description
This project focuses on classifying dog breeds using deep learning techniques. The goal is to identify the breed of a dog from an image using a deep neural network. The dataset contains 120 different breeds, making this a fine-grained image classification problem. This work is based on a subset of the ImageNet dataset and involves training a model to distinguish between similar-looking breeds.<br>

# Dataset
The dataset used for this project is a subset of ImageNet, containing images of 120 dog breeds.<br>
It includes a limited number of images per class, which adds to the challenge of the classification task.<br>
Images are provided in high resolution, requiring preprocessing and augmentation for optimal model performance.<br>

# Model Overview
Model Architecture: MobileNet V2<br>
Transfer Learning: The model is fine-tuned on the dog breed dataset starting from a pretrained MobileNet V2, which has been trained on the larger ImageNet dataset.<br>
Why MobileNet V2?: It is a lightweight model with a good balance between accuracy and computational efficiency, making it suitable for real-time applications.<br>

# Workflow
Data Preparation: Image preprocessing, augmentation, and batching.<br>
Model Training: Fine-tuning the MobileNet V2 model with early stopping and tensorboard monitoring.<br>
Evaluation: Measuring model accuracy on validation data and identifying performance metrics.<br>
Prediction: Generating breed predictions with confidence scores for new images.<br>
Visualization: Analyzing correct and incorrect predictions to improve the model.<br>

# Download the dataset:
The dataset can be downloaded from Kaggle - https://www.kaggle.com/c/dog-breed-identification/data.<br>


# Usage
Training the Model:<br>
Use the provided training script to train the model on the dataset.<br>
Making Predictions:<br>
Utilize the prediction script to classify new images.<br>
Evaluation:<br>
Run the evaluation script to see the model's performance on test data.<br>
Results<br>
The model achieves 65% accuracy on the validation set, demonstrating its ability to distinguish between 120 dog breeds even with just using 1000 images (Due to time restriction).<br>
Further improvements can be made through hyperparameter tuning and experimenting with other architectures.<br>
# Future Work
Explore other architectures like EfficientNet and ResNet.<br>
Enhance the dataset using data augmentation techniques.<br>
Deploy the model as a web application for real-time dog breed identification.<br>
# License
This project is licensed under the MIT License - see the LICENSE file for details.<br>

# Acknowledgments
Kaggle for providing the dataset.<br>
The deep learning community for their continuous contributions to open-source projects and research.<br>
