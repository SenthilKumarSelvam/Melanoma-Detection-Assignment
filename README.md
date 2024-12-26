# Melanoma Skin Cancer Detection
> Develop a custom CNN in TensorFlow to detect melanoma, a deadly skin cancer, using a dataset of 2,357 images from ISIC. The model aims to assist dermatologists by automating diagnosis and reducing manual effort.


## Table of Contents
* [General Info](#general-information)
* [About Dataset](#Data-understanding)
* [Approaches](#Building,-Training,-and-Evaluating-a-CNN-Model-for-Multi-Class-Classification-with-Data-Augmentation-and-Class-Imbalance-Handling)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- The dataset, sourced from the International Skin Imaging Collaboration (ISIC), consists of 2,357 images of both malignant and benign oncological conditions. Images are categorized based on the ISIC classification system. While the dataset is balanced across most classes, melanomas and moles have a slightly higher representation.
- The objective of this assignment is to build a CNN-based model capable of accurately detecting melanoma. Melanoma is a severe form of cancer responsible for 75% of skin cancer-related deaths. Early detection is critical for effective treatment. A reliable model that evaluates images and alerts dermatologists about the presence of melanoma can significantly reduce the manual effort required for diagnosis.
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->
## Data understanding
- The dataset comprises a total of 2,357 images, representing malignant and benign oncological conditions. These images were obtained from the International Skin Imaging Collaboration (ISIC), a trusted source for dermatological image data..
- Key details about the dataset:
-   Source: International Skin Imaging Collaboration (ISIC).
-   Content: Images of malignant and benign skin conditions.
-   Classification: All images were categorized based on ISIC's classification system.
-   Distribution: The dataset is balanced, with subsets containing an equal number of images.
-   ## Sample Pictorial Representation of Skin Types
  ![image](https://github.com/user-attachments/assets/704bbadf-198a-40a2-a3fe-098cb0afc2e2)
-   ## Sample Pictorial Representation of Skin Types
  ![image](https://github.com/user-attachments/assets/c863fe2c-f455-4b82-8e52-6e37a3bc0c57)

## Building,-Training,-and-Evaluating-a-CNN-Model-for-Multi-Class-Classification-with-Data-Augmentation-and-Class-Imbalance-Handling
- ## Model Approach for Skin Cancer Classification using Skin Lesion Images
- Rescaling Layer: Normalizes image pixel values from [0, 255] to [0, 1].
- Convolutional Layer: Applies convolution to reduce image size and extract features.
- Pooling Layer: Reduces spatial dimensions and computational complexity.
- Dropout Layer: Randomly sets input units to zero during training to prevent overfitting.
- Flatten Layer: Converts multi-dimensional data into a one-dimensional array for the classifier.
- Dense Layer: Fully connects neurons from previous layers to learn complex patterns.
- Activation Function (ReLU): Applies ReLU activation to help the model learn faster and avoid vanishing gradients.
- Activation Function (Softmax): Converts the final outputs into probabilities for each class in multiclass classification.
- Data Augmentation: Applies random transformations (rotation, scaling, flipping) to increase training data diversity and improve generalization.
- Normalization: Scales pixel values to the range [0, 1] using Rescaling(1./255) to stabilize training and accelerate convergence.
- Convolutional Layers (3 layers): Sequentially applies 3 convolutional layers with increasing filter counts (16, 32, 64) and ReLU activations, with 'same' padding to preserve spatial dimensions.
- Pooling Layers: Max-pooling layers follow each convolution to downsample feature maps and retain important information, reducing complexity.
- Dropout Layer (0.2 rate): Prevents overfitting by randomly dropping neurons after the last max-pooling layer.
- Flatten Layer: Flattens 2D feature maps into a 1D vector for input into the fully connected layers.
- Fully Connected Layers: Two dense layers (128 neurons and output layer) with ReLU activations, followed by the final classification output.
- Output Layer: The number of neurons corresponds to the number of target labels, outputting classification probabilities.
- Model Compilation: Compiles the model with Adam optimizer, Sparse Categorical Crossentropy loss, and accuracy as the evaluation metric.
- Training: Trains the model for 50 epochs using the fit method, with callbacks for ModelCheckpoint and EarlyStopping to prevent overfitting and improve model convergence.

## Conclusions
- ## Conclusion 1 from the analysis of the model-1
  ![image](https://github.com/user-attachments/assets/24a4f3b4-8d27-49b5-8a54-848717853ce3)
- ## Observations:
-   The model achieves high training accuracy (~90%) but struggles with validation accuracy, which fluctuates around 55-80%, indicating overfitting.
-   Training loss decreases steadily, while validation loss fluctuates after an initial decline.
-   The performance gap between training and validation confirms the model lacks generalization.
-   Overfitting is likely due to limited and less diverse training data.
- ## Solutions:
-   To address this, data augmentation, regularization (dropout, L2), early stopping, and batch normalization are recommended to improve generalization.
  
- ## Conclusion 2 from the analysis of the model-2
  ![image](https://github.com/user-attachments/assets/2d150751-8e96-4079-8295-f016a9a478a1)

- ## Observations:
-   
- ## Solutions:
- ## Conclusion 3 from the analysis of the model-3
  ![image](https://github.com/user-attachments/assets/24a4f3b4-8d27-49b5-8a54-848717853ce3)
- ## Observations:
-   
- ## Solutions:
- ## Conclusion 4 from the analysis of the model-4
  ![image](https://github.com/user-attachments/assets/24a4f3b4-8d27-49b5-8a54-848717853ce3)
- ## Observations:
-  
- ## Solutions:

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python - version 3.11.4
- Matplotlib - version 3.7.1
- Numpy - version 1.24.3
- Pandas - version 1.5.3
- Seaborn - version 0.12.2
- Tensorflow - version 2.17.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Contact
Created by [@senthilkumarselvam] - feel free to contact me!
