# celebrity_image_classification


The code given is for a Deep Learning model using TensorFlow and Keras to classify images of five different celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli. The script covers various aspects of the typical workflow in building, training, and evaluating a Convolutional Neural Network (CNN) for image classification.

 1. Data Loading and Preprocessing:
- Image directories for each celebrity are specified.
- Images are loaded, resized to (224, 224), and converted to NumPy arrays.
- Labels (0 to 4) are assigned based on the celebrity class.

 2. Dataset Information:
- The length of image data for each celebrity is printed.

 3. Train-Test Split:
- The dataset is split into training and testing sets using `train_test_split()` from scikit-learn.

4. Normalization:
- The pixel values of the images are normalized to the range [0, 1] by dividing the floating point of each value by 255.

 5. Model Definition:
- A Sequential model is defined using Keras.
- Convolutional layers (Conv2D) with "ReLU" activation and max-pooling layers are stacked.
- A flatten layer is added to transition from convolutional layers to dense layers.
- Dense layers with 'ReLU' activation are added, and the final layer uses softmax activation for multi-class classification.
- The model summary is printed.

 6. Model Compilation:
- The model is compiled with the 'Adam' optimizer and 'sparse_categorical_crossentropy'  loss for multi-class classification.
- Accuracy is chosen as the metric.

 7. Model Training:
- The model is trained on the training dataset for 10 epochs with a batch size of 64.
- Training history is stored in the `history` variable.

 8. Prediction Function:
- A function `make_prediction` is defined to make predictions on new images.
- The function takes an image file path, preprocesses the image, and predicts the class.

 9. Image Prediction:
- Image paths of sample images for each celebrity are provided.
- The model predicts the class label for each image using the `make_prediction` function.
- Predictions are printed for each image.

In conclusion, the image classification code deploys a convolutional neural network for recognizing celebrities from images. To further enhance the model performance, future iterations may explore alternative architectures, fine-tune hyperparameters and ensure a balanced distribution of images among different classes to mitigate biases. Regular updates and retraining with new data can help the model to adapt to evolving patterns and maintain optimal accuracy.
