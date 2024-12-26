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
-   ## No. of images belonging to a Type v/s Types of Skin Lesion
  ![image](https://github.com/user-attachments/assets/c863fe2c-f455-4b82-8e52-6e37a3bc0c57)

## Model development and its observation
- ## Model 1: Developing a CNN Model for Accurate 9-Class Classification with Pixel Value Normalization Using Rescaling Layer
- This CNN model is designed for 9-class image classification. It normalizes pixel values to the [0, 1] range using a Rescaling layer and consists of three convolutional layers with ReLU activation and max-pooling, followed by a flattening layer and two dense layers (128 units and 9 output classes).
- ## Observations:
   ![image](https://github.com/user-attachments/assets/24a4f3b4-8d27-49b5-8a54-848717853ce3)
-   Improvement in accuracy: Training accuracy increases from 35.28% to 95.84% over 20 epochs, showing the model is learning well.
-   Decreasing training loss: Training loss drops from 1.6900 to 0.0963, indicating effective model convergence.
-   Validation performance: Validation accuracy improves from 48.91% to 87.98%, though it fluctuates in the middle epochs.
-   Validation loss: Validation loss initially decreases but fluctuates later, suggesting potential overfitting.
-   Overfitting potential: The model may be overfitting, as training accuracy keeps rising while validation accuracy plateaus or drops slightly in later epochs.

- ## Model 2: Developing a CNN Model for Robust Classification with Data Augmentation, Dropout Regularization, and Pixel Value Normalization Using Rescaling Layer
- ## Observations:
  ![image](https://github.com/user-attachments/assets/1bc87221-d84f-4a86-8456-641fb6f19d32)
- Improvment in Accuracy: Both training and validation accuracy improve over time (training: 34.76% to 83.51%, validation: 41.70% to 85.58%).
- Decreasing training loss: Training and validation loss decrease consistently, indicating better predictions (training: 1.7039 to 0.4470, validation: 1.5982 to 0.4471).
- Validation Performance: Validation accuracy lags initially but exceeds training accuracy in later epochs, showing good generalization.
- Validation loss: Accuracy plateaus around epoch 14-15, indicating diminishing returns in learning.
- Potential Overfitting: Some validation loss fluctuations suggest slight overfitting, though overall validation performance remains stable.

- ## Data Treatment
- Analyzing Class Distribution in Training Data for Imbalance Detection
- ![image](https://github.com/user-attachments/assets/ecf3b29a-0294-45f7-86c6-4f5fa9206671)
- Imbalance: The dataset is imbalanced, with varying class proportions. A clear class imbalance is evident within the training data.
- Top Classes:"Pigmented benign keratosis," "melanoma," and "basal cell carcinoma" have the highest proportions. Pigmented benign keratosis" and "melanoma" significantly outweigh other classes, representing approximately 20.63%and 19.56% of the data, respectively.
- Minor Classes: "Seborrheic keratosis" and others are underrepresented. The class "seborrheic keratosis" comprises the smallest proportion of samples, accounting for approximately 3.44%.
- Model Bias: Imbalance may cause the model to favor larger classes, affecting smaller ones.

-  Rectify the class imbalance through the Augmentor
-![image](https://github.com/user-attachments/assets/69ecd520-2294-4de6-a17a-44bf674e0c48)
- Observations: Class Distribution Before Augmentation:
- Significant imbalance, with top classes (e.g., pigmented benign keratosis) at 20.63% and lowest (seborrheic keratosis) at 3.44%. Class Distribution After Augmentation:
- More balanced proportions, with the lowest class now at 10.02% and top classes reduced to around 12.47%. Overall Improvement:
- Enhanced balance reduces bias, improving the model's ability to generalize across all classes.

- Improvements:
- Reduction in Imbalance: Maximum class proportion decreased from 20.63% to 12.47%, and minimum increased from 3.44% to 10.02%.
- Increased Representation of Minority Classes: Underrepresented classes now have a more substantial presence, aiding feature learning.
- Potential for Improved Model Performance: Expect better accuracy and metrics across all classes due to reduced bias.
- Mitigation of Overfitting: Balanced dataset helps prevent overfitting to dominant classes, leading to a more robust model.
- Outcome: Augmentation effectively addressed class imbalance, enhancing dataset equity and expected model performance.

- ## Model 3: Developing a CNN Model with Data Augmentation and Regularization for Image Classification
- ## Observations:
- ![image](https://github.com/user-attachments/assets/ad14b1e1-2150-49f3-b4d4-b9063fb1427d)
- Improvement in Accuracy: Both training and validation accuracy improve over time (training: 34.21% to 77.71%, validation: 24.25% to 58.25%).
- Decreasing Training Loss: Training loss decreases consistently, indicating better predictions (training: 1.8878 to 0.5918, validation: 10.5293 to 1.2856).
- Validation Performance: Validation accuracy shows fluctuations but peaks at 73.34% in Epoch 17, indicating some good generalization.
- Validation Loss: Validation loss exhibits spikes, particularly in Epochs 5 and 10, suggesting instability in learning.
- Potential Overfitting: The gap between training and validation accuracy indicates potential overfitting, with validation performance not improving significantly in later epochs.

- ## Model 4: Developing a CNN Model for Image Classification with Data Augmentation, Dropout Regularization and without batch normalization
- ## Observations:
- ![image](https://github.com/user-attachments/assets/065e937e-f2a6-4a78-8b44-7f54f4afacc1)
- Initial Performance: The model started with low accuracy (37.09% training, 44.42% validation) in the first epoch.
- Improvement Over Epochs: Both training and validation accuracy improved consistently, reaching 86.01% and 87.35% respectively by epoch 20.
- Validation Accuracy Trends: Validation accuracy improved in most epochs but plateaued or decreased in some (epochs 7, 10, 11, 12, 15, 17, 20), indicating potential overfitting.
- Loss Reduction: Training loss decreased steadily from 1.6277 to 0.3715, while validation loss showed some fluctuations but generally trended downward.
- Overfitting Signs: The gap between training and validation accuracy suggests the model may be overfitting, especially in later epochs.
- Best Model Saving: The model was saved whenever validation accuracy improved, with a peak of *87.35% *in epoch 19.
- Final Evaluation: The final validation accuracy of 84.87% in epoch 20 indicates potential overfitting; testing on a separate dataset is recommended.
- Future Considerations: Techniques like early stopping, regularization, or data augmentation could help mitigate overfitting and improve generalization.

- ## Model Evaluation on a Test Image: Actual vs. Predicted Class"
- ## Observations:
- ![image](https://github.com/user-attachments/assets/34518140-dd61-421f-8b3c-7c39d741f20a)
- Model Performance: The model correctly identified the test image as "basal cell carcinoma."
- Accuracy: The prediction matches the actual class, indicating high accuracy for this instance.
- Confidence in Predictions: The correct classification suggests the model has learned relevant features, though confidence levels are not provided.
- In summary, the model successfully identified the test image as "basal cell carcinoma," indicating effective learning and classification for this specific case. However, further evaluation is necessary to understand its performance comprehensively.

- ## Conclusion
-  ## Observations:
-  ![image](https://github.com/user-attachments/assets/cfe70e8e-f4a4-49f8-8777-775c90210413)
- ## Model Performance:
- Model 1 achieved the highest training accuracy (95.84%) and validation accuracy (87.98%), indicating it performed best among the four models.
- Model 4 also performed well, with a final validation accuracy of 84.87% and a best validation accuracy of 87.35%. It had the lowest final training loss (0.3715) among the models.
- Model 2 and Model 3 had lower performance, with final validation accuracies of 85.58% and 58.25%, respectively. Model 3 had the lowest overall performance, with a final validation accuracy of only 58.25%.
- ## Overfitting:
- Model 1 shows signs of overfitting, as the validation accuracy fluctuated and did not consistently improve after reaching its peak. This is evident from the drop in validation accuracy in later epochs.
- Model 4, while also showing improvement, maintained a more stable validation accuracy throughout the epochs.
- ## Generalization:
- Model 1's high training accuracy suggests it may have learned the training data well, but the validation accuracy indicates it may not generalize as effectively as Model 4.
- Model 4's performance suggests a better balance between training and validation accuracy, indicating it may generalize better to unseen data.
- In summary, while Model 1 performed the best in terms of accuracy, Model 4 demonstrated a more stable performance and may be more suitable for generalization. Model 2 and Model 3 lagged behind in both training and validation metrics, indicating they may require further tuning or adjustments.


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
