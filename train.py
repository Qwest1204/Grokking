from data.dataset import ModularAdditionDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.model import GrokkingTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR

DIM = 97
BATCH_SIZE = 16
MODEL_DIM = 128
VOCAB = 98
D_FF = 512
DROPOUT = 0.1
NUM_HEAD = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = ModularAdditionDataset(DIM)
train, test = torch.utils.data.random_split(dataset, [3000, 6409])
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
scheduler = CosineAnnealingLR(optimizer, T_max=10000)

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image
import torchvision.transforms as transforms


def setup_tensorboard():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/experiment_{current_time}')
    return writer


def compute_loss_landscape(model, criterion, dataloader, device,
                           steps=20, range_scale=1):
    """
    Вычисляет лосс-ландшафт вокруг текущих параметров модели
    """
    model.eval()

    # Получаем батч данных для оценки
    data_iter = iter(dataloader)
    batch_y, batch_x = next(data_iter)
    batch_y = batch_y.to(device)
    batch_x = batch_x.to(device)

    # Сохраняем исходные параметры
    original_params = {name: param.clone() for name, param in model.named_parameters()
                       if param.requires_grad}

    # Выбираем два случайных направления
    directions = {}
    for name, param in original_params.items():
        # Создаем случайные направления (norm=1)
        d1 = torch.randn_like(param)
        d2 = torch.randn_like(param)
        d1 = d1 / torch.norm(d1)
        d2 = d2 / torch.norm(d2)
        directions[name] = (d1, d2)

    # Создаем сетку для оценки
    x = np.linspace(-range_scale, range_scale, steps)
    y = np.linspace(-range_scale, range_scale, steps)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    with torch.no_grad():
        for i in range(steps):
            for j in range(steps):
                alpha = X[i, j]
                beta = Y[i, j]

                # Применяем смещение к параметрам
                for name, param in model.named_parameters():
                    if name in original_params:
                        d1, d2 = directions[name]
                        param.data = original_params[name] + alpha * d1 + beta * d2

                # Вычисляем лосс
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y.squeeze(1))
                Z[i, j] = loss.item()

    # Восстанавливаем исходные параметры
    for name, param in model.named_parameters():
        if name in original_params:
            param.data = original_params[name]

    return X, Y, Z


def plot_loss_landscape_3d(X, Y, Z):
    """Создает 3D визуализацию лосс-ландшафта"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Создаем поверхность
    surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                           linewidth=0, antialiased=True,
                           alpha=0.8, rstride=1, cstride=1)

    # Добавляем контурные линии на поверхность
    ax.contour(X, Y, Z, 10, offset=Z.min(), cmap='coolwarm')

    # Настройки графика
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape')

    # Добавляем цветовую шкалу
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Преобразуем в изображение для TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # Преобразуем в тензор для TensorBoard
    image = Image.open(buf)
    image_tensor = transforms.ToTensor()(image)

    return image_tensor


def plot_loss_landscape_2d(X, Y, Z):
    """Создает 2D тепловую карту лосс-ландшафта"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Создаем контурный график
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)

    # Добавляем точку в центре (текущая позиция модели)
    ax.plot(0, 0, 'ro', markersize=10, label='Current position')

    # Настройки графика
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_title('Loss Landscape (Contour)')
    ax.legend()

    # Добавляем цветовую шкалу
    plt.colorbar(contour, ax=ax)

    # Преобразуем в изображение для TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # Преобразуем в тензор для TensorBoard
    image = Image.open(buf)
    image_tensor = transforms.ToTensor()(image)

    return image_tensor


def compute_gradient_norm(model):
    """Вычисляет нормы градиентов для мониторинга"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def train_with_logging(transformer, dataloader_train, dataloader_test,
                       optimizer, criterion, num_epochs=100, device='cuda',
                       log_loss_landscape_every=10, loss_landscape_steps=55):
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

    def log_loss_landscape(epoch):
        """Логирование лосс-ландшафта"""
        try:
            print(f"\nComputing loss landscape for epoch {epoch}...")

            # Вычисляем лосс-ландшафт
            X, Y, Z = compute_loss_landscape(
                transformer, criterion, dataloader_train,
                device, steps=loss_landscape_steps, range_scale=100.0
            )

            # Создаем визуализации
            landscape_3d = plot_loss_landscape_3d(X, Y, Z)
            landscape_2d = plot_loss_landscape_2d(X, Y, Z)

            # Логируем в TensorBoard
            writer.add_image('loss_landscape/3d_view', landscape_3d, epoch)
            writer.add_image('loss_landscape/2d_contour', landscape_2d, epoch)

            # Логируем дополнительные метрики
            min_loss = np.min(Z)
            max_loss = np.max(Z)
            avg_loss = np.mean(Z)

            writer.add_scalar('loss_landscape/min_loss', min_loss, epoch)
            writer.add_scalar('loss_landscape/max_loss', max_loss, epoch)
            writer.add_scalar('loss_landscape/avg_loss', avg_loss, epoch)
            writer.add_scalar('loss_landscape/range', max_loss - min_loss, epoch)

            print(f"Loss landscape computed: min={min_loss:.4f}, max={max_loss:.4f}")

        except Exception as e:
            print(f"Error computing loss landscape: {e}")

    step = 0
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        transformer.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(dataloader_train, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_idx, (y, x) in enumerate(train_bar):
            y = y.to(device)
            x = x.to(device)

            optimizer.zero_grad()
            logits, attentions = transformer(x)

            loss = criterion(logits, y.squeeze(1))
            loss.backward()

            # Логируем норму градиентов
            grad_norm = compute_gradient_norm(transformer)
            writer.add_scalar('train/gradient_norm', grad_norm, step)

            optimizer.step()

            _, predicted = torch.max(logits, 1)
            train_total += y.size(0)
            train_correct += (predicted == y.squeeze(1)).sum().item()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                writer.add_scalar('train/loss_step', loss.item(), step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step)
                writer.add_scalar('train/batch_accuracy',
                                  100 * (predicted == y.squeeze(1)).sum().item() / y.size(0), step)

                if batch_idx == 0:
                    log_attention_maps(attentions, epoch, prefix="train")

            step += 1
            train_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(dataloader_train)
        train_accuracy = 100 * train_correct / train_total

        writer.add_scalar('train/loss_epoch', avg_train_loss, epoch)
        writer.add_scalar('train/accuracy', train_accuracy, epoch)

        # Логирование лосс-ландшафта (реже, чтобы не замедлять обучение)
        if epoch % log_loss_landscape_every == 0:
            log_loss_landscape(epoch)

        if epoch % 10 == 0:
            log_model_weights(transformer, epoch)

        # Валидация
        transformer.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            test_bar = tqdm(dataloader_test, desc=f'Epoch {epoch + 1}/{num_epochs} [Test]')
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

        # Сохраняем лучшую модель

        writer.add_scalar('metrics/train_test_loss_gap', avg_train_loss - avg_test_loss, epoch)
        writer.add_scalar('metrics/accuracy_gap', train_accuracy - test_accuracy, epoch)
        writer.add_scalar('metrics/best_test_accuracy', best_test_acc, epoch)

        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        print(f'  Best Test Acc: {best_test_acc:.2f}%')
        print(f'  Gradient Norm: {grad_norm:.4f}')
        print('-' * 60)

    writer.close()
    print("\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print("\nTo view results, run:")
    print("tensorboard --logdir=runs/")

    return transformer


# Дополнительная функция для анализа лосс-ландшафта после обучения
def analyze_loss_landscape(model, dataloader, criterion, device,
                           save_path='loss_landscape_analysis.png'):
    """
    Детальный анализ лосс-ландшафта после обучения
    """
    model.eval()

    # Вычисляем лосс-ландшафт с разными масштабами
    scales = [0.01, 0.05, 0.1, 0.2]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, scale in enumerate(scales):
        X, Y, Z = compute_loss_landscape(
            model, criterion, dataloader, device,
            steps=25, range_scale=scale
        )

        ax = axes[idx]
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.plot(0, 0, 'ro', markersize=10)
        ax.set_xlabel('Direction 1')
        ax.set_ylabel('Direction 2')
        ax.set_title(f'Loss Landscape (scale={scale})')

        # Добавляем цветовую шкалу для каждого подграфика
        plt.colorbar(contour, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Loss landscape analysis saved to {save_path}")

trained_model = train_with_logging(
        transformer=transformer,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=10000,
        device=DEVICE
    )

analyze_loss_landscape(
    trained_model,
    dataloader_train,
    criterion,
    device=DEVICE,
    save_path='final_loss_landscape.png'
)