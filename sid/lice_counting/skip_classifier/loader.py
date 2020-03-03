from torchvision import datasets, transforms
import torch
import os


DATA_DIR = 'data_dir'

def load_dataset():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_dataset = datasets.ImageFolder(DATA_DIR, transform)
    
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)
    dataset_size = len(image_dataset)
    class_names = image_dataset.classes
    print(f'Found Classes: {class_names}')
    print(f'dataset size: {dataset_size}')
    # TODO: ensure that the gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    return dataloader

def show_images():
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    
    # Get a batch of training data
    inputs, classes = next(iter(dataloader))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    
    imshow(out, title=[class_names[x] for x in classes])

if __name__ == '__main__':
    dataloader = load_dataset()

