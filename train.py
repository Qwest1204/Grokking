from data.dataset import AdditionalModule

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.model import GrokkingTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR

DIM = 97
BATCH_SIZE = 16
MODEL_DIM = 128
VOCAB = 125
D_FF = 512
DROPOUT = 0.1
NUM_HEAD = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = AdditionalModule(DIM)
train, test = torch.utils.data.random_split(dataset, [5000, 4409])
dataloader_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

transformer = GrokkingTransformer(d_model=MODEL_DIM,
                                  d_ff=D_FF,
                                  num_heads=NUM_HEAD,
                                  dropout=DROPOUT,
                                  vocab_size=VOCAB,
                                  ).to(DEVICE)

optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-9
    )

criterion = nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=100)

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime

def setup_tensorboard():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/experiment_{current_time}')
    return writer

def train_with_logging(transformer, dataloader_train, dataloader_test,
                       optimizer, criterion, num_epochs=100, device='cuda'):

    writer = setup_tensorboard()

    def log_model_weights(model, step):
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'weights/{name}', param.data, step)
                if param.grad is not None:
                    writer.add_histogram(f'grads/{name}', param.grad, step)

    def log_attention_maps(attentions, step, prefix="train"):
        if attentions is not None:
            for head_idx in range(attentions.shape[1]):
                attention_map = attentions[0, head_idx].detach().cpu()
                writer.add_image(
                    f'{prefix}/attention_head_{head_idx}',
                    attention_map.unsqueeze(0),
                    step
                )

    step = 0

    for epoch in range(num_epochs):
        transformer.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (y, x) in enumerate(train_bar):
            y = y.to(device)
            x = x.to(device)

            optimizer.zero_grad()
            logits, attentions = transformer(x)

            loss = criterion(logits, y.squeeze(1))
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits, 1)
            train_total += y.size(0)
            train_correct += (predicted == y.squeeze(1)).sum().item()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                writer.add_scalar('train/loss_step', loss.item(), step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step)

                if batch_idx == 0:
                    log_attention_maps(attentions, epoch, prefix="train")

            step += 1
            train_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(dataloader_train)
        train_accuracy = 100 * train_correct / train_total

        writer.add_scalar('train/loss_epoch', avg_train_loss, epoch)
        writer.add_scalar('train/accuracy', train_accuracy, epoch)

        if epoch % 10 == 0:
            log_model_weights(transformer, epoch)

        transformer.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            test_bar = tqdm(dataloader_test, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for batch_idx, (y, x) in enumerate(test_bar):
                y = y.to(device)
                x = x.to(device)

                logits, attentions = transformer(x)
                loss = criterion(logits, y.squeeze(1))

                test_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                test_total += y.size(0)
                test_correct += (predicted == y.squeeze(1)).sum().item()

                if batch_idx == 0:
                    log_attention_maps(attentions, epoch, prefix="test")

        avg_test_loss = test_loss / len(dataloader_test)
        test_accuracy = 100 * test_correct / test_total

        writer.add_scalar('test/loss_epoch', avg_test_loss, epoch)
        writer.add_scalar('test/accuracy', test_accuracy, epoch)

        writer.add_scalar('metrics/train_test_gap', avg_train_loss - avg_test_loss, epoch)
        writer.add_scalar('metrics/accuracy_gap', train_accuracy - test_accuracy, epoch)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        print('-' * 50)

    writer.close()
    print("tensorboard --logdir=runs/")

    return transformer

trained_model = train_with_logging(
        transformer=transformer,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=100,
        device=DEVICE
    )