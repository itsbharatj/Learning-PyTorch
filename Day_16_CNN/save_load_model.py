import torch

class save_load: 

    def save_model(model, optimizer, epoch, loss, filepath):
        """
        Save model checkpoint including architecture, parameters, optimizer state, and training info
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, filepath)
        print(f'Model saved to {filepath}')

    def load_model(model, optimizer, filepath, device):
        """
        Load saved model checkpoint
        """
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        return model, optimizer, epoch, loss