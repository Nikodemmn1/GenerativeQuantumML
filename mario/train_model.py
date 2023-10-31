from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from EarlyStopper import EarlyStopper
import torch
import torch.nn as nn
from QLSTMario import LSTMArio
from QLSTMario import n_qubits, str_entangled_layers
from MarioDataset import MarioDataset
from torch.utils.data import DataLoader
from parse_preprocessed_data import get_inputs_and_targets
from sklearn.model_selection import train_test_split


def main():
    print("Num GPUs Available: ", torch.cuda.device_count())

    hidden_size = 128
    learning_rate = 2e-3
    dropout = 0.2
    batch_size = 300
    val_batch_size = 1024
    num_layers = 1
    max_epochs = 20
    patience = 5
    validation_prop = 0.2
    seq_length = 200

    # DATA LOADING

    _, _, vocab_size, inputs, targets = get_inputs_and_targets('data_preprocessed/mario.txt', seq_length)
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, targets, test_size=validation_prop)
    train_dataset = MarioDataset(inputs_train, targets_train)
    val_dataset = MarioDataset(inputs_val, targets_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    # MODEL AND OTHER ELEMENTS

    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    model = LSTMArio(vocab_size, hidden_size, num_layers, dropout, quantum_dummy=False).cuda()
    loss_function = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()

    # MODEL TRAINING

    for epoch in range(max_epochs):
        train_running_loss = 0.0
        train_batches = 0
        train_correct_preds = 0

        model.train(True)
        for x, y_true in tqdm(train_dataloader):
            optimizer.zero_grad()

            y_pred, _ = model(x.cuda())
            y_true = y_true.cuda()

            #loss = loss_function(y_pred.reshape(y_pred.size(0) * y_pred.size(1), -1), y_true.flatten())
            loss = loss_function(y_pred.swapaxes(1, 2), y_true)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            train_batches += 1

            train_correct_preds += torch.sum(torch.argmax(y_pred, 2) == y_true)

        val_running_loss = 0.0
        val_batches = 0
        val_correct_preds = 0

        model.eval()
        with torch.no_grad():
            for x, y_true in tqdm(val_dataloader):
                y_pred, _ = model(x.cuda())
                y_true = y_true.cuda()
                loss = loss_function(torch.swapaxes(y_pred, 1, 2), y_true)
                val_running_loss += loss.item()
                val_batches += 1
                val_correct_preds += torch.sum(torch.argmax(y_pred, 2) == y_true)

        train_loss = train_running_loss / train_batches
        val_loss = val_running_loss / val_batches
        train_acc = train_correct_preds / targets_train.size
        val_acc = val_correct_preds / targets_val.size
        print(f"Epoch {epoch} - train loss: {train_loss} - val loss: {val_loss} - "
              f"train acc: {train_acc} - val acc: {val_acc}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)
        writer.flush()

        if early_stopper.early_stop(val_loss):
            break

        torch.save(model, f'trained_models/mario_lstm_seqlen={seq_length}_nqubits={n_qubits}_'
                          f'strent={str_entangled_layers}_epoch={epoch}.pt')
    writer.close()

if __name__ == '__main__':
    main()
