# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1077" height="883" alt="image" src="https://github.com/user-attachments/assets/12acd421-e7eb-4dce-bfc2-44352028fe20" />

## DESIGN STEPS

### STEP 1:
Load the dataset, clean it by handling missing values, drop irrelevant columns, encode categorical variables, and normalize features.

### STEP 2:
Split the data into training and testing sets.
### STEP 3:
Build a neural network model with multiple layers using PyTorch.
### STEP 4:
Train the model using CrossEntropyLoss and Adam optimizer.
### STEP 5:
Evaluate the model with accuracy, confusion matrix, and classification report.
### STEP 6:
Test the model with new sample data for prediction.

## PROGRAM

### Name: NISHMA SHERIN .K
### Register Number: 212224240104

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)



    def forward(self, x):
         x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x

        

```
```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
     for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print("Name: NISHMA SHERIN .K  ")    
print("Register No: 212224240104")     
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Prediction for a sample input
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name: NISHMA SHERIN .K  ")    
print("Register No: 212224240104")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')

```



## Dataset Information

<img width="1336" height="261" alt="Screenshot 2026-02-11 160811" src="https://github.com/user-attachments/assets/5e6d5caf-7bb5-4a84-be69-a01be979abc8" />



## OUTPUT

<img width="963" height="612" alt="Screenshot 2026-02-11 160911" src="https://github.com/user-attachments/assets/6686ff10-f7c1-42cd-8079-13078dc3a28c" />




### Confusion Matrix
### Classification Report


<img width="596" height="460" alt="Screenshot 2026-02-11 160843" src="https://github.com/user-attachments/assets/b94e35e2-4373-4fe4-8350-f61ac4951bc1" />


### New Sample Data Prediction


<img width="426" height="117" alt="Screenshot 2026-02-11 160919" src="https://github.com/user-attachments/assets/170f67d0-e41d-4d5d-96f4-c6ca6db5b43f" />


## RESULT
The program to develop a neural network regression model for the given dataset has been successfully executed.
