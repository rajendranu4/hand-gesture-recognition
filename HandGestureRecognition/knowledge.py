import torch
import torch.nn as tnn
import os

import global_constants as gc
from label_rename import convert_idx_to_label
from classification_cnn import CNN_2D
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Learning:
    def __init__(self, cnn_model):
        self.model = cnn_model
        self.criterion = tnn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(cnn_model.parameters(), lr=gc.LEARNING_RATE)

    def train(self, dataloader):
        if os.path.isfile(gc.BASE_PATH + "\\" + "gesture_model_cnn.pt"):
            print("\nThere is a model saved which is already trained with the same dataset")
            #self.model = torch.load(torch.load(gc.BASE_PATH + "\\" + "gesture_model_cnn.pt"))
            self.model.load_state_dict(torch.load(gc.BASE_PATH + "\\" + "gesture_model_cnn.pt"))
            return

        losses = []

        for epoch in range(1, gc.N_EPOCHS + 1):
            print("Epoch.... {}".format(epoch))
            # keep-track-of-training-and-validation-loss
            loss_epoch = 0.0

            # training-the-model
            self.model.train()
            for data, target in dataloader:
                # clear-the-gradients-of-all-optimized-variables
                self.optimizer.zero_grad()
                # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
                output = self.model(data)

                # calculate-the-batch-loss
                #output = output.type(torch.LongTensor)
                target = target.type(torch.LongTensor)
                loss = self.criterion(output, target)
                # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
                loss.backward()
                # perform-a-single-optimization-step (parameter-update)
                self.optimizer.step()
                # update-training-loss
                loss_epoch += loss.item() * data.size(0)

            # calculate-average-losses
            loss_epoch = loss_epoch / len(dataloader.sampler)
            losses.append(loss_epoch)

            # print-training/validation-statistics
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, loss_epoch))

        plt.plot(range(gc.N_EPOCHS), losses)
        plt.title('Epoch VS Loss')
        plt.xlabel('EPOCHS')
        plt.ylabel('LOSS')
        plt.show()

        torch.save(self.model.state_dict(), gc.BASE_PATH + "\\" + "gesture_model_cnn.pt")
        #torch.save(self.model, gc.BASE_PATH + "\\" + "gesture_model_cnn.pt")

    def test(self, dataloader):
        # Switch the model to evaluation mode (so we don't backpropagate or drop)
        self.model.eval()
        loss = 0
        correct = 0

        with torch.no_grad():
            batch_count = 0
            for data, target in dataloader:
                batch_count += 1

                # Get the predicted classes for this batch
                output = self.model(data)

                #output = output.type(torch.LongTensor)
                target = target.type(torch.LongTensor)

                # Calculate the loss for this batch
                loss += self.criterion(output, target).item()

                # Calculate the accuracy for this batch
                _, predicted = torch.max(output.data, 1)
                #_, target = torch.max(target.data, 1)
                print("Original")
                print(target)
                print("Predicted")
                print(predicted)
                correct += torch.sum(target == predicted).item()

        # Calculate the average loss and total accuracy for this epoch
        avg_loss = loss / batch_count
        print('Test set: Average loss: {:.2f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, correct, len(dataloader.dataset),
            100. * correct / len(dataloader.dataset)))

    def test_single_known(self, X, y):
        # Switch the model to evaluation mode (so we don't backpropagate or drop)
        self.model.eval()

        with torch.no_grad():

            # Get the predicted classes for this batch
            output = self.model(X[None, ...])

            output = output.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)

            # Calculate the accuracy for this batch
            predicted = torch.argmax(output)
            y = torch.argmax(y)

            print("\nOriginal Label: {}".format(convert_idx_to_label(y.item())))
            print("Predicted Label: {}".format(convert_idx_to_label(predicted.item())))

    def test_single_unknown(self, X):
        self.model.eval()

        with torch.no_grad():

            # Get the predicted classes for this batch
            output = self.model(X[None, ...])
            #output = self.model(X)

            output = output.type(torch.FloatTensor)

            # Calculate the accuracy for this batch
            predicted = torch.argmax(output)
            predicted_prob = torch.amax(output)

            return convert_idx_to_label(predicted.item()), predicted_prob