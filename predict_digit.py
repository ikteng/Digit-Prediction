import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_data_loader(training = True):
    custom_transform=transforms.Compose([
        transforms.ToTensor(), #converts a PIL Image or numpy.ndarray to tensor
        transforms.Normalize((0.1307,), (0.3081,)) #normalizes the tensor with a mean and standard deviation which goes as the two parameters respectively
        ])

    # set training set and testing set
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=custom_transform) # contains image & label to train neural network
    test_set = datasets.MNIST(root="./data", train=False, transform=custom_transform) # contains images & labels for model evaluation

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)

    if(training==True):
        return train_loader
    else:
        return test_loader

# build model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            #28 x 28 = 784
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10) # 10 classes for digits 0-9
            )
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, criterion, T):
    # set optimizer
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # set model to training mode
    model.train()

    for epoch in range(T):
        running_loss=0.0
        correct_pred=0
        total_pred=0
        for i, data in enumerate(train_loader,0):
            inputs,labels=data
            opt.zero_grad()
            outputs=model(inputs)

            loss=criterion(outputs,labels)

            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()*labels.size(0)
            _,predictions=torch.max(outputs,1)

            # collect the correct predictions for each batch
            for label, prediction in zip(labels,predictions):
                if label==prediction:
                    correct_pred+=1
                total_pred+=1

        accuracy=(correct_pred/total_pred)*100
            
        print(f'Train Epoch {epoch}: Accuracy: {correct_pred}/{total_pred} ({accuracy :.2f}%) Loss: {running_loss/total_pred :.3f}')

# def predict_digit(model,image_path):
#     # sets the model to evaluation model
#     model.eval()

#     # transformation applied to images loaded from the file
#     preprocess = transforms.Compose([
#         transforms.Resize((28, 28)), # resize image
#         transforms.Grayscale(num_output_channels=1), # convert to grayscale
#         transforms.ToTensor(), # transform to PyTorch tensor
#         transforms.Normalize((0.1307,), (0.3081,)) # normalize using mean and standard deviation
#     ])

#     # open the image from the image_path
#     image = Image.open(image_path)
#     # transform to Pytorch tensor
#     image_tensor = preprocess(image).unsqueeze(0)

#     with torch.no_grad():
#         output = model(image_tensor) # computes the output of neural network given the input image
#         _, predicted = torch.max(output, 1) # maximum values, predicted class index for the input image

#     return predicted.item()

def predict_digit(model,image):
    # sets the model to evaluation model
    model.eval()

    # transformation applied to images loaded from the file
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)), # resize image
        transforms.Grayscale(num_output_channels=1), # convert to grayscale
        transforms.ToTensor(), # transform to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize using mean and standard deviation
    ])

    # open the image from the image_path
    # image = Image.open(image_path)
    # transform to Pytorch tensor
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor) # computes the output of neural network given the input image
        _, predicted = torch.max(output, 1) # maximum values, predicted class index for the input image

    return predicted.item()

def digit_segmentation(image_path):
    model.eval()

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_regions = []
    for contour in contours:
        # Get bounding box coordinates for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Exclude very small contours (noise)
        if cv2.contourArea(contour) > 100:
            digit_regions.append((x, y, w, h))

    return digit_regions

def adjust_colors_and_predict(image_path):
    model.eval()

    # Detect individual digit regions in the image
    digit_regions = digit_segmentation(image_path)

    # Sort digit regions from left to right based on x-coordinate
    digit_regions.sort(key=lambda x: (x[0], x[1]))

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Iterate through detected digit regions
    for region in digit_regions:
        x, y, w, h = region
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with bounding boxes around the detected digit regions
        cv2.imshow('Digit Regions', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Crop the digit region from the image
        digit_image = image[y:y + h, x:x + w]

        # Convert the image to grayscale using OpenCV
        # grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)

        # Calculate average pixel value to determine the dominant color
        avg_pixel_value = np.mean(grayscale_image)

        # Check if the image has a predominantly light background (dark font)
        if avg_pixel_value > 128:
            # Invert colors if it's a light background (dark font)
            inverted_image = cv2.bitwise_not(grayscale_image)

            # Convert the inverted NumPy array to a PIL image
            pil_inverted_image = Image.fromarray(inverted_image)

            # Use the PIL image for transformations
            predicted_digit = predict_digit(model, pil_inverted_image)
            print(f"The predicted digit in the image is: {predicted_digit}")
        else:
            # Convert the original image to a PIL image
            pil_image = Image.fromarray(image)

            # Use the PIL image for transformations
            predicted_digit = predict_digit(model, pil_image)
            print(f"The predicted digit in the image is: {predicted_digit}")

# save the model for future use
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

if __name__ == '__main__':
    # get dataloaders
    train_loader=get_data_loader()
    test_loader=get_data_loader(False)
    
    # train your model
    model=NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    # train_model(model,train_loader,criterion,20)
    model.load_state_dict(torch.load('Digits Recognition\mnist_digit_classifier.pth')) # -> if you have already train the model and saved

    model.eval()
    # predict the digit
    # Provide the path to your image
    image_path = "Digits Recognition\img_5.png"

    # Call the function to adjust colors and predict the digit
    adjust_colors_and_predict(image_path)

    # save the trained model to a file
    # save_model(model, 'Digits Recognition\mnist_digit_classifier.pth') 
