import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from typing import Union


class TeacherStudents:
    def __init__(self, input_size=784, num_students=3):
        # input size
        self.input_size = input_size
        # number of students
        self.num_students = num_students
        # device: whether gpu or cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Teacher network
        self.teacher_network = Net(input_size=input_size, hidden_size=100).to(self.device)
        # Students networks
        self.students_networks = nn.ModuleList()
        for i in range(num_students):
            self.students_networks.append(Net(input_size=input_size, hidden_size=100).to(self.device))
        # define Teacher optimizer
        self.teacher_optimizer = torch.optim.Adam(self.teacher_network.parameters(), lr=1e-5, weight_decay=1e-3)
        # define Students optimizers
        self.students_optimizers = []  # list and not ModuleList, ModuleList only works for submodules of layers
        for i in range(num_students):
            self.students_optimizers.append(torch.optim.Adam(self.students_networks[i].parameters(), lr=1e-6, weight_decay=1e-3))

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Train Teacher and Students together
        :param x:
        :param y:
        """

        # set networks to train mode
        self.teacher_network.train()
        for student in self.students_networks:
            student.train()

        # move data to tensors
        x = torch.from_numpy(x.to_numpy())
        y = torch.from_numpy(y.to_numpy())

        # define dataset and loader
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Train the model
        print('NOWW')
        for epoch in range(1000):
            for i_batch, (x, y) in enumerate(dataloader):
                # move batch to device
                # Debugging: Check the shapes of x and y
                # print(f"x shape: {x.shape}")
                # print(f"y shape before unsqueeze: {y.shape}")

                # Debugging: Check the values in y
                # Ensure y values are within the range [0, 1]
                # if not ((y >= 0) & (y <= 1)).all():
                #     raise ValueError("Target values in `y` must be between 0 and 1 for BCELoss")

                x, y = x.float().to(self.device), y.float().unsqueeze(1).to(self.device)  # y is transformed to (batch_size, 1)

                # forward step for Teacher
                teacher_hidden_outs, teacher_p = self.teacher_network(x)
                students_hidden_outs = []
                students_p = []

                # forward step for each Student
                for student in self.students_networks:
                    hidden_outs, p = student(x)
                    students_hidden_outs.append(hidden_outs)
                    students_p.append(p)

                # zero grads
                self.teacher_optimizer.zero_grad()
                for optimizer in self.students_optimizers:
                    optimizer.zero_grad()

                # combined loss for Teacher and Students
                first_loss = torch.tensor(0.0).to(self.device)
                # loss for each layer and each student
                for student_i in range(self.num_students):
                    for layer_j in range(len(teacher_hidden_outs)):

                        # Debugging: Check values in teacher_hidden_outs and students_hidden_outs
                        if not ((teacher_hidden_outs[layer_j] >= 0) & (teacher_hidden_outs[layer_j] <= 1)).all():
                            print(f"Invalid values in teacher_hidden_outs at layer {layer_j}")
                        if not ((students_hidden_outs[student_i][layer_j] >= 0) & (
                                students_hidden_outs[student_i][layer_j] <= 1)).all():
                            print(f"Invalid values in students_hidden_outs for student {student_i} at layer {layer_j}")
                            print(f"students_hidden_outs[student_i][layer_j]: {students_hidden_outs[student_i][layer_j]}")

                        loss = nn.BCELoss()(teacher_hidden_outs[layer_j], students_hidden_outs[student_i][layer_j])
                        first_loss += loss
                first_loss /= (self.num_students * len(teacher_hidden_outs))

                second_loss = torch.tensor(0.0).to(self.device)
                # loss for each student output
                for student_i in range(self.num_students):

                    # Debugging: Check values in teacher_p and students_p
                    if not ((teacher_p >= 0) & (teacher_p <= 1)).all():
                        print(f"Invalid values in teacher_p")
                    if not ((students_p[student_i] >= 0) & (students_p[student_i] <= 1)).all():
                        print(f"Invalid values in students_p for student {student_i}")

                    loss = nn.BCELoss()(teacher_p, students_p[student_i])
                    second_loss += loss

                second_loss /= self.num_students
                # loss for teacher

                # Debugging: Check values in teacher_p and y
                if not ((teacher_p >= 0) & (teacher_p <= 1)).all():
                    print(f"Invalid values in teacher_p")
                if not ((y >= 0) & (y <= 1)).all():
                    print(f"Invalid values in y")

                loss = nn.BCELoss()(teacher_p, y)
                # combined_loss += loss
                combined_loss = first_loss + second_loss + loss

                # backward step
                combined_loss.backward()
                self.teacher_optimizer.step()

                first_student = self.students_networks[0]

                for student, optimizer in zip(self.students_networks, self.students_optimizers):
                    # Gradient clipping
                    # torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    optimizer.step()

                # print(f"Epoch: {epoch}, Iteration: {i_batch}, Loss: {combined_loss.item()}")
                if epoch % 10 == 0 and i_batch == 0:
                    print(f"Epoch: {epoch}, Iteration: {i_batch}, Loss: {loss.item()}")


    def predict(self, x: Union[pd.DataFrame, np.ndarray]):

        # set networks to eval mode
        self.teacher_network.eval()
        for student in self.students_networks:
            student.eval()

        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.from_numpy(x.to_numpy()).float()
            else:
                x = torch.from_numpy(x).float()
            x = x.to(self.device)

            _, p = self.teacher_network(x)

            predicted = p > 0.5
            predicted = predicted.cpu().detach().numpy().astype(int)
            # print(f"Predicted: {predicted}")
        return predicted

    def predict_proba(self,
                      x: Union[pd.DataFrame, np.ndarray]):

        # set networks to eval mode
        self.teacher_network.eval()
        for student in self.students_networks:
            student.eval()

        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.from_numpy(x.to_numpy()).float()
            else:
                x = torch.from_numpy(x).float()
            x = x.to(self.device)

            _, p = self.teacher_network(x)

            # TODO: check dimensions of p, check concatenation of 1-p and p
            # print(f"p shape: {p.shape}")
            p = p.cpu().detach().numpy()
            # print(f"p: {p}")
            proba = np.concatenate([1 - p, p], axis=1)
            # print(f"proba: {proba}")
            # print(f"proba shape: {proba.shape}")
            # print(f"Predicted: {predicted}")
        return proba


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers=1, num_classes=1):
        super(Net, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # list of hidden layers
        self.hidden_layers = nn.ModuleList()
        # list of hidden activations
        self.hidden_activations = nn.ModuleList()
        # List of output layers for hidden layers
        self.output_layers = nn.ModuleList()

        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_activations.append(nn.ReLU())
            self.output_layers.append(nn.Linear(hidden_size, num_classes))

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_last = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Debugging: Check input values
        if torch.isnan(x).any():
            print("NaN values found in input")
        if torch.isinf(x).any():
            print("Inf values found in input")
        # first layer
        x = self.fc1(x)
        x = self.relu(x)

        # Debugging: Check values after first layer
        if torch.isnan(x).any():
            print("NaN values found after fc1 and relu")

        # perform hidden layers and save the outputs
        hidden_outs = []
        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)  # Linear layer
            x = self.hidden_activations[i](x)  # Activation function
            hidden_outs.append(x)  # Save the output of the hidden layer

        # last layer
        x = self.fc_last(x)
        p = self.sigmoid(x)

        hiddens_probs = []
        for i in range(self.num_hidden_layers):
            output_layer = self.output_layers[i](hidden_outs[i])  # Get the output of hidden layer and pass through Linear layer
            hiddens_probs.append(self.sigmoid(output_layer))  # perform Sigmoid and save

        return hiddens_probs, p
