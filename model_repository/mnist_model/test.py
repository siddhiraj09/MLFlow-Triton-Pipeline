import mlflow
import mlflow.onnx
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)  # Updated this line
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(self.conv2(x), 2)  # Add max pooling layer
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Training settings
batch_size = 64
epochs = 1
lr = 0.01
momentum = 0.5
log_interval = 10

# Initialize data loaders
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# Initialize MLflow experiment
mlflow.set_experiment("MNIST_ONNX_Experiment")

with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("momentum", momentum)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                mlflow.log_metric('loss', loss.item(), step=epoch * len(train_loader) + batch_idx)

        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        mlflow.log_metric('test_loss', test_loss, step=epoch)
        mlflow.log_metric('accuracy', accuracy, step=epoch)

    # Convert and log the model in ONNX format
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(model, dummy_input, "mnist_model.onnx")
    mlflow.onnx.log_model(onnx_model=onnx.load("mnist_model.onnx"), artifact_path="mnist_model")

print('Training complete.')

# Testing the logged ONNX model
session = onnxruntime.InferenceSession("mnist_model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

correct = 0
total = 0
for data, target in test_loader:
    data = data.numpy()
    inputs = {input_name: data}
    outputs = session.run([output_name], inputs)
    pred = np.argmax(outputs[0], axis=1)
    correct += np.sum(pred == target.numpy())
    total += target.size(0)

accuracy = 100. * correct / total
print(f'Accuracy of the ONNX model on the test images: {accuracy:.2f}%')
