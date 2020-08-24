import torch
import torchvision
import torch.optim as optim
from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb

transforms = {
    'train': dense_transforms.Compose(
        [
         dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
         dense_transforms.RandomHorizontalFlip(),
         dense_transforms.ToTensor(),
         dense_transforms.ToHeatmap(radius = 2)]),
    
    'test': dense_transforms.Compose(
        [dense_transforms.ToTensor(),
         dense_transforms.ToHeatmap(radius = 2)])
    }
    


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector().to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'fcn.th')))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    criterion = FocalLoss()
    
    train_data = load_detection_data('dense_data/train', num_workers=2, batch_size=args.batch_size, transform = transforms['train'])
    
    valid_data = load_detection_data('dense_data/valid', num_workers=2, batch_size=args.batch_size, transform = transforms['test'])

    global_step = 0 
    
    for epoch in range(args.num_epoch):
        print("epoch " + str(epoch))
        model.train()
        train_loss = 0
        valid_loss = 0

        for index, (image, peak, size) in enumerate(train_data):
            image, peak = image.to(device), peak.to(device)
            logits = model(image)

            loss = criterion(logits, peak)
            print(loss.size())
            train_loss += loss
            
            if train_logger and global_step % 100 == 0:
                train_logger.add_scalar('loss', loss, global_step=global_step)
                print(f'Batch: {index}, Training Loss: {loss}, Average Loss: {train_loss / index + 1}')

                with torch.no_grad():
                    vis_index = 100
                    vis_img, vis_peaks, vis_size = valid_data.dataset[vis_index]
                    vis_img = vis_img[None,]
                    vis_img, vis_peaks, vis_size = vis_img.to(device), vis_peaks.to(device), vis_size.to(device)
                    vis_logits = model(vis_img)
                    heatmap_prob = vis_logits.sigmoid()
                    image = torch.cat([vis_img[0], heatmap_prob[0], vis_peaks], 2).detach().cpu()
                    train_logger.add_image('image', (torchvision.utils.make_grid(image, padding=5, pad_value=1) * 255).byte(), global_step=global_step)
                

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        model.eval()

        with torch.no_grad():
            for index, (image, peak, size) in enumerate(valid_data):
                image, peak = image.to(device), peak.to(device)
                logits = model(image)
                
                loss = criterion(logits, peak)
                valid_loss += loss

            scheduler.step(loss)
            
            if valid_logger and global_step % 100 == 0:
                valid_logger.add_scalar('loss', loss, global_step)
                print(f'Batch: {index}, Valid Loss: {loss}, Average Loss: {valid_loss / index + 1}')
            
        save_model(model)
        


import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        loss = nn.BCEWithLogitsLoss(reduction='none')
        BCE_loss = loss(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return torch.mean(F_loss)

    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-b', '--batch_size', type=int,default=128)
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    args = parser.parse_args()
    train(args)
