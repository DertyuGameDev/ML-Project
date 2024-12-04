import torch


def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    """Сохранение чекпоинта"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved at epoch {epoch} with loss {loss:.4f}')

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """Загрузка чекпоинта"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Checkpoint loaded from epoch {epoch} with loss {loss:.4f}')
    return epoch, loss