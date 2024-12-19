import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import Model3
from utils import get_data_loaders, save_sample_grid
from torchsummary import summary
from tabulate import tabulate
import time
from torch.utils.tensorboard import SummaryWriter
import os

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def print_summary(model, train_accs, test_accs, total_params, best_acc):
    """Print training summary with all metrics"""
    print("\n" + "="*70)
    print(" "*25 + "TRAINING SUMMARY")
    print("="*70)
    
    # Model metrics table
    headers = ["Metric", "Value"]
    table_data = [
        ["Total Parameters", f"{total_params/1000:.1f}k"],
        ["Best Test Accuracy", f"{best_acc:.2f}%"],
        ["Final Train Accuracy", f"{train_accs[-1]:.2f}%"],
        ["Final Test Accuracy", f"{test_accs[-1]:.2f}%"],
        ["Learning Rate", "0.01"],
        ["Momentum", "0.9"],
        ["Batch Size", "128"],
        ["Total Epochs", "100"]
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Epoch-wise accuracies table
    print("\nEpoch-wise Accuracies:")
    print("-"*70)
    
    epoch_headers = ["Epoch", "Train Accuracy", "Test Accuracy"]
    epoch_data = []
    
    for epoch, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs), 1):
        epoch_data.append([epoch, f"{train_acc:.2f}%", f"{test_acc:.2f}%"])
    
    print(tabulate(epoch_data, headers=epoch_headers, tablefmt="grid"))
    print("="*70)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, target)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    return 100*correct/processed

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(test_loader.dataset)
    test_acc.append(accuracy)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy

def main():
    # Create runs directory for logs
    if not os.path.exists('runs'):
        os.makedirs('runs')
    
    # Create directory for sample images
    if not os.path.exists('sample_images'):
        os.makedirs('sample_images')
    
    # Initialize tensorboard
    writer = SummaryWriter(f'runs/CIFAR10_{time.strftime("%Y%m%d_%H%M%S")}')
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("CUDA Available?", use_cuda)
    
    # First get data without transforms
    print("\nSaving original images...")
    train_loader_no_transforms, _ = get_data_loaders(
        batch_size=128,
        apply_transforms=False  # Get data without transforms
    )
    
    # Save images before augmentation
    save_sample_grid(
        dataset=train_loader_no_transforms.dataset,
        transforms=None,
        prefix="before_augmentation"
    )
    
    # Now get data with transforms
    print("\nSaving augmented images...")
    train_loader, test_loader = get_data_loaders(
        batch_size=128,
        apply_transforms=True  # Get data with transforms
    )
    
    # Save images after augmentation
    save_sample_grid(
        dataset=train_loader.dataset,
        transforms=train_loader.dataset.transform,
        prefix="after_augmentation"
    )
    
    model = Model3().to(device)
    summary(model, input_size=(3, 32, 32))
    
    BATCH_SIZE = 128
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    best_train_acc = 0
    best_test_acc = 0
    best_test_epoch = 0
    total_params = sum(p.numel() for p in model.parameters())
    
    EPOCHS = 100
    for epoch in range(EPOCHS):
        print(f"\nEPOCH: {epoch+1}")
        train_accuracy = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, test_loader)
        
        # Log to tensorboard
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        
        # Update best accuracies
        best_train_acc = max(best_train_acc, train_accuracy)
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_test_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')

    # Print final summary
    print_summary(
        model=model,
        train_accs=train_acc,
        test_accs=test_acc,
        total_params=total_params,
        best_acc=best_test_acc
    )
    
    writer.close()

if __name__ == '__main__':
    main() 