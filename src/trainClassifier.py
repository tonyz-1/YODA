from datetime import datetime
from matplotlib import pyplot as plt

import torch


def train(model, n_epochs, train_dl, device, optimizer, scheduler, loss_fn, weightPath):
    model.train()
    losses_list = []
    test_loss = []
    for epoch in range(n_epochs):
        loss_epoch = 0.0
        index = 0
        for imgs, labels in train_dl:
            index += 1
            input_ims = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # output = model(input_ims)
            # loss = loss_fn(output, labels)

            output = model.forward(input_ims)
            loss = loss_fn(output, labels)

            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            if index % 100 == 0:
                print(f"Batch {index} loss: {loss}")

        model.train()
        losses_list.append(loss_epoch / index)
        print(f"Epoch {epoch} loss: {losses_list[epoch]}")
        if epoch % 5 == 0:
            state_dict = model.state_dict()
            torch.save(state_dict, weightPath)
            #test_a, test_l = functions.accuracy_loss(model, test_dl, device, loss_fn)
            #print(f"Test accuracy epoch {epoch}: {test_a}")
            #test_accuracy.append(test_a)
            #test_loss.append(test_l)
            #model.train()
        scheduler.step()

    plt.plot(losses_list, label="Train Loss")
    plt.plot(test_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
