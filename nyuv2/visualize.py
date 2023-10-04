import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

COLORS = torch.tensor(
[
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (128, 0, 0),     # Maroon
    (0, 128, 0),     # Green (Dark)
    (0, 0, 128),     # Navy
    (128, 128, 0),   # Olive
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (128, 128, 128)  # Gray
]
)


def onehot_segmentation_to_img(onehot, colors=COLORS):
    indices = torch.argmax(onehot, dim=1)
    return indices_segmentation_to_img(indices, colors)


def indices_segmentation_to_img(indices, colors=COLORS):
    indices = indices.long()
    if indices.size(1) == 1:
        # Remove single channel axis.
        indices = indices[:, 0]
    rgbs = colors[indices]
    return rgbs


def random_crop(img, label, depth, h=128, w=256):
    height, width = img.shape[-2:]
    i = random.randint(0, height - h)
    j = random.randint(0, width - w)
    img = img[:, i: i + h, j: j + w]
    label = label[i: i + h, j: j + w]
    depth = depth[:, i: i + h, j: j + w]
    return img, label, depth



from data import NYUv2
from city2scape.models import SegNetMtan

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = NYUv2(root="data", train=False, augmentation=False)

fimtl_ckpt = "/scratch/hvp2011/implement/FS-MTL/city2scape/outputs/moo-sam/data_path=PosixPathdata/n_epochs=200/batch_size=8/lr=0.0002/seed=0/model=mtan/method=imtl/adaptive=True/rho=0.0003/apply_augmentation=True/c=0.4/190.pth"
imtl_ckpt = "/scratch/hvp2011/implement/FS-MTL/nash_checkpoint/city2scape/outputs/nash-mtl/data_path=PosixPath/scratch/hvp2011/implement/FS-MTL/city2scape/data/n_epochs=200/batch_size=2/method=imtl/lr=0.0001/method_params_lr=0.025/gpu=0/seed=42/nashmtl_optim_niter=20/update_weights_every=1/main_task=0/c=0.4/dwa_temp=2.0/model=mtan/apply_augmentation=True/wandb_project=None/wandb_entity=None/190.pth"


flat_net = SegNetMtan()
flat_net.load_state_dict(torch.load(fimtl_ckpt, map_location=torch.device('cpu')))
erm_net = SegNetMtan()
erm_net.load_state_dict(torch.load(imtl_ckpt, map_location=torch.device('cpu')))
flat_net = flat_net.to(device)
erm_net = erm_net.to(device)

for idx in tqdm(range(100)):
    image, semantic, depth, _ = dataset[idx]

    image, semantic, depth = random_crop(image, semantic, depth)
    print(image.shape, semantic.shape, depth.shape)
    gt_semantic = indices_segmentation_to_img(semantic, COLORS)
    gt_semantic = gt_semantic.numpy().astype('uint8')
    print(gt_semantic.shape)
    # gt_semantic = Image.fromarray(gt_semantic)
    # gt_semantic.save(f'plot/semantic_{idx}.png')
    rgb_image = (255*image).permute(1,2,0).numpy().astype('uint8')
    # rgb_image = Image.fromarray(rgb_image)
    # rgb_image.save(f'plot/image_{idx}.png')


    image = image.unsqueeze(0).to(device)
    semantic = semantic.unsqueeze(0).to(device)
    depth = depth.unsqueeze(0).to(device)
    
    with torch.no_grad():
        flat_segmatic, flat_depth = flat_net(image)
        erm_segmatic, erm_depth = erm_net(image)

    flat_segmatic = flat_segmatic.squeeze().permute(1,2,0).cpu().argmax(dim=-1)
    erm_segmatic = erm_segmatic.squeeze().permute(1,2,0).cpu().argmax(dim=-1)
    
    print("SHAPE", flat_segmatic.shape, erm_segmatic.shape)
    
    flat_segmatic = indices_segmentation_to_img(flat_segmatic, COLORS)
    flat_segmatic = flat_segmatic.numpy().astype('uint8')
    erm_segmatic = indices_segmentation_to_img(erm_segmatic, COLORS)
    erm_segmatic = erm_segmatic.numpy().astype('uint8') 
    fig = plt.figure(figsize=(10, 7))
 
    fig.add_subplot(1, 4,1)
    
    # showing image
    plt.imshow(rgb_image, interpolation='nearest')
    plt.axis('off')
    plt.title("Image")
    
    # Adds a subplot at the 1st position
    fig.add_subplot(1, 4, 2)
    
    # showing image
    plt.imshow(gt_semantic, interpolation='nearest')
    plt.axis('off')
    plt.title("Ground Truth")
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 4, 3)
    
    # showing image
    plt.imshow(erm_segmatic)
    plt.axis('off')
    plt.title("ERM")
    
    # Adds a subplot at the 3rd position
    fig.add_subplot(1, 4, 4)
    
    # showing image
    plt.imshow(flat_segmatic)
    plt.axis('off')
    plt.title("Ours")

    plt.savefig(f'plot/{idx}.png')

    # cmap = plt.cm.jet
    # cmap.set_bad(color="black")
    depth0 = depth[0]
    depth0 = depth0/10
    # plt.imshow(depth0)
    # plt.axis('off')
    # plt.savefig(f'plot/depth_{idx}.png')
    