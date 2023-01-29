import torch
import torch.nn as tnn
import torch.nn.functional as tnf


class CNN_2D(tnn.Module):
    def __init__(self):
        super(CNN_2D, self).__init__()
        self.conv1 = tnn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)

        # A second convolutional layer takes 12 input channels, and generates 24 outputs
        self.conv2 = tnn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

        # We in the end apply max pooling with a kernel size of 2
        self.pool = tnn.MaxPool2d(kernel_size=2)

        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = tnn.Dropout2d(p=0.2)

        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # This means that our feature tensors are now 32 x 32, and we've generated 24 of them

        # We need to flatten these in order to feed them to a fully-connected layer
        self.fc1 = tnn.Linear(in_features=24 * 32 * 32, out_features=8)

    def forward(self, x):
        # In the forward function, pass the data through the layers we defined in the init function

        # Use a ReLU activation function after layer 1 (convolution 1 and pool)
        x = tnf.relu(self.pool(self.conv1(x)))

        # Use a ReLU activation function after layer 2
        x = tnf.relu(self.pool(self.conv2(x)))

        # Select some features to drop to prevent overfitting (only drop during training)
        x = tnf.dropout(self.drop(x), training=self.training)

        # Flatten
        x = x.view(-1, 24 * 32 * 32)
        # Feed to fully-connected layer to predict class
        x = self.fc1(x)
        # Return class probabilities via a log_softmax function
        #return torch.sigmoid(x)
        return x