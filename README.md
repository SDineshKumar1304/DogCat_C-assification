
# Dog and Cat Classification using DeepLearning
- Implementing a deep learning pipeline for classifying images of cats and dogs using the PyTorch framework. It begins by preparing a dataset, visualizing class distributions, and splitting data into training and testing sets. Image preprocessing involves resizing, random transformations, and normalization.

- The model architecture utilizes a pretrained ResNet-18 with the final layer adapted for binary classification (cats vs. dogs). Training involves defining a loss function (CrossEntropyLoss), an optimizer (Adam), and a learning rate scheduler. The training loop iterates through epochs, evaluating performance with training loss plotted over epochs.

- After training, the model is evaluated on a test set to compute accuracy, achieving approximately 90.27% accuracy in this case. Trained model parameters are saved for future use. Inference capabilities include loading a single image, preprocessing it for prediction, and using the model to classify it as a cat or dog, with visualization of the prediction.

- The code's modularity, utilizing PyTorch's DataLoader for efficient batch processing and GPU acceleration if available. Visualization tools such as matplotlib and seaborn aid in understanding dataset characteristics and model performance. Overall, it provides a robust framework for training and deploying a deep learning model for binary image classification tasks.

## References:
https://www.geeksforgeeks.org/cat-dog-classification-using-convolutional-neural-network-in-python/
https://pytorch.org/vision/stable/index.html
http://pytorch.org/vision/stable/models.html
## Support:
- Mr.Kanav Bansal Cheif DataScientist (Innomatics Research Labs)
# DataSet
https://huggingface.co/datasets/microsoft/cats_vs_dogs
