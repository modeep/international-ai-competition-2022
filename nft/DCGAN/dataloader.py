import torch
from torchvision import datasets
from torchvision import transforms

def get_dataloader(batch_size,
                   image_size,
                   data_dir,
                   num_workers=0):

    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # create transformer to transform images
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),  # resize
                                    transforms.ToTensor(),  # convert to tensor
                                    transforms.Normalize(*stats)])  # normalize to be between -1 and 1

    # create the dataset
    dataset = datasets.ImageFolder(root=data_dir,
                                   transform=transform)

    # create the dataloader
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True)

    return data_loader