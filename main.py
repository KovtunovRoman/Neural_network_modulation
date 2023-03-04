import torch
import torch.onnx
from sklearn import utils
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import Data
from helper_function import plot_decision_boundary

""" Set the hyperparameters for data creation """
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

""" Create multi-class data"""

X_blob, y_blob = utils.shuffle(Data.X_blob, Data.y_blob)
print(X_blob, y_blob)

X_blob = torch.Tensor(X_blob).type(torch.float)
y_blob = torch.Tensor(y_blob).type(torch.LongTensor)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED
                                                                        )

""" Create device agnostic code """

device = "cuda" if torch.cuda.is_available() else "cpu"

""" Build model """


class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=14):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features),  # how many classes are there?
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


""" Create an instance of BlobModel and send it to the target device """

model_4 = BlobModel(input_features=NUM_FEATURES,
                    output_features=NUM_CLASSES,
                    hidden_units=8).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), lr=0.005)

y_logits = model_4(X_blob_test.to(device))

""" Perform softmax calculation on logits across dimension 1 to get prediction probabilities """

y_pred_probs = torch.softmax(y_logits, dim=1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model_4.state_dict(), path)


print(model_4(X_blob_train.to(device))[0].shape, NUM_CLASSES)
torch.manual_seed(42)

epochs = 25000

""" Put data to target device """

X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(X_blob_train)  # model outputs raw logits
    y_pred = torch.softmax(y_logits, dim=1).argmax(
        dim=1)  # go from logits -> prediction probabilities -> prediction labels

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    """ Testing """

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_pred)

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

""" Make predictions """

model_4.eval()
with torch.inference_mode():
    print(X_blob_test)
    y_logits = model_4(X_blob_test)
    print(model_4(X_blob_test))

""" Turn predicted logits in prediction probabilities """

y_pred_probs = torch.softmax(y_logits, dim=1)

""" Turn prediction probabilities into prediction labels """

y_preds = y_pred_probs.argmax(dim=1)

""" Compare first 10 model preds and test labels """

print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()

saveModel()


