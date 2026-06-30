from torchvision import transforms

def get_rgb_transforms(normalize=False, rgb_mean=None, rgb_std=None):

    train_steps = [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ]

    eval_steps = [
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ]

    if normalize==True:
        train_steps.append(
        transforms.Normalize(mean=rgb_mean, std=rgb_std))

        eval_steps.append(
        transforms.Normalize(mean=rgb_mean, std=rgb_std))

    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose(eval_steps)

    return train_transform, eval_transform


def get_ms_transforms(normalize=False, ms_mean=None, ms_std=None):
    
    train_steps = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ]

    eval_steps = []

    if normalize==True:
        train_steps.append(
        transforms.Normalize(mean=ms_mean, std=ms_std))

        eval_steps.append(
        transforms.Normalize(mean=ms_mean, std=ms_std))

    train_ms_transform = transforms.Compose(train_steps)
    eval_ms_transform = transforms.Compose(eval_steps)

    return train_ms_transform, eval_ms_transform