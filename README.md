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
print("Name: NAUSHEEN FATHIMA A")
print("Register No: 212224230179")     
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
print("Name: NAUSHEEN FATHIMA A ")    
print("Register No: 212224230179")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')

```



## Dataset Information

<img width="1385" height="331" alt="Screenshot 2026-02-11 103518" src="https://github.com/user-attachments/assets/64a9fba8-1587-49c3-b4a9-6a8c11a06545" />


## OUTPUT
<img width="889" height="628" alt="Screenshot 2026-02-11 103414" src="https://github.com/user-attachments/assets/c40b1687-63e4-4745-b35e-d6734f42b538" />




### Confusion Matrix

<img width="321" height="177" alt="Screenshot 2026-02-11 103459" src="https://github.com/user-attachments/assets/f99c3c13-4cc8-4945-b272-c96b3f5ebef9" />

### Classification Report

<img width="612" height="267" alt="Screenshot 2026-02-11 103503" src="https://github.com/user-attachments/assets/b20977fb-8454-40a6-87e0-ccc15c57fd27" />



### New Sample Data Prediction

<img width="644" height="143" alt="Screenshot 2026-02-11 103421" src="https://github.com/user-attachments/assets/7e4927d7-6e21-4f5c-835e-e90a33de4698" />


## RESULT
The program to develop a neural network regression model for the given dataset has been successfully executed.
