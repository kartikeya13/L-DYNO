import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

# Dataset class
class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(os.listdir(os.path.join(root_dir, 'image_2')))
        self.gt_poses = np.loadtxt('00.txt')

    def __len__(self):
        return len(self.image_paths) - 1

    def __getitem__(self, idx):
        img1 = Image.open(os.path.join(self.root_dir, 'image_2', self.image_paths[idx]))
        img2 = Image.open(os.path.join(self.root_dir, 'image_2', self.image_paths[idx + 1]))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        gt_rel_pose = self.gt_poses[idx]
        return img1, img2, torch.from_numpy(gt_rel_pose).float()

# Pose Estimation LSTM class
class PoseEstimationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PoseEstimationLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
# Attention Layer class
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        attention_weights = self.attention_network(features)
        attended_features = features * attention_weights
        return attended_features, attention_weights

# Siamese Network class
class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, attention_dim):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.attention_layer = AttentionLayer(input_size, attention_dim)
        self.pose_estimation = PoseEstimationLSTM(input_size, hidden_size, num_layers, output_size)

    def forward(self, img1, img2):
        feat1, feat2 = self.feature_extractor(img1), self.feature_extractor(img2)
        print("Feature shape:", feat1.shape)
        feat1_flat = feat1.view(feat1.size(0), -1)
        feat2_flat = feat2.view(feat2.size(0), -1)
        
        attended_feat1, attention_weights1 = self.attention_layer(feat1_flat)
        attended_feat2, attention_weights2 = self.attention_layer(feat2_flat)
        
        feats = torch.stack([attended_feat1, attended_feat2], dim=1)
        pose_params = self.pose_estimation(feats)
        return pose_params, attended_feat1, attended_feat2

# Data transforms
data_transforms = transforms.Compose([
    transforms.Resize((1240,376)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = KITTIDataset('kitti2/', transform=data_transforms)
train_dataloader = DataLoader(train_dataset,  batch_size=1, num_workers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Loss function
def pose_loss_fn(predicted_pose, gt_pose):
    translation_loss = torch.norm(predicted_pose[:,:3] - gt_pose[:,:3],dim=1)
    rotation_loss = torch.norm(predicted_pose[:, 3:] - gt_pose[:, 3:],dim=1)
    loss = torch.mean( translation_loss + rotation_loss)
    return loss

# Load the trained model
def load_trained_model(model_path, input_size, hidden_size, num_layers, output_size, attention_dim):
    model = SiameseNetwork(input_size, hidden_size, num_layers, output_size, attention_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Add map_location=device
    return model

model_path = 'newcolor_att.pth'
attention_dim = 512  # Specify the attention_dim used in training
input_size = 256 * 77 * 23
hidden_size = 512
num_layers = 2
output_size = 6
trained_model = load_trained_model(model_path, input_size, hidden_size, num_layers, output_size, attention_dim)

# Visualize attention weights
def find_output_shape(input_shape, feature_extractor):
    test_input = torch.randn(1, *input_shape)
    output = feature_extractor(test_input)
    return output.shape
output_shape = find_output_shape((3, 1240, 376), trained_model.feature_extractor)
print("Output shape:", output_shape)

def visualize_attended_features_overlay(image, attended_features, attended_shape):
    attended_features = attended_features.view(attended_shape).cpu().detach().numpy()
    attended_features_mean = np.mean(attended_features, axis=0)
    
    threshold = np.percentile(attended_features_mean, 85) # You can adjust the threshold to get more or fewer points
    attended_points = np.where(attended_features_mean > threshold)
    
    print("att_points",attended_points)
    
    img_with_dots = np.transpose(image.cpu().numpy(), (1, 2, 0)).copy()
    img_with_dots = (img_with_dots * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img_with_dots = np.clip(img_with_dots, 0, 255).astype(np.uint8)

    for y, x in zip(*attended_points):
        scaled_y, scaled_x = (y / attended_shape[1] * img_with_dots.shape[0], x / attended_shape[2] * img_with_dots.shape[1])
        cv2.circle(img_with_dots, (int(scaled_x), int(scaled_y)), 2, (255, 0, 0), -1)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_with_dots)
    plt.title('Attended Features Overlay with Red Dots')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(attended_features_mean, cmap='gray')
    plt.title('Attention Map')
    plt.axis('off')

    plt.show()

image1, image2, gt_rel_pose = next(iter(train_dataloader))
image1, image2 = image1[0].to(device), image2[0].to(device)
pose_params, attended_feat1, attended_feat2 = trained_model(image1.unsqueeze(0), image2.unsqueeze(0))
print("att_feat1",attended_feat1)
attended_shape = output_shape[1:] 

# Save directory
def save_attended_features_overlay(image, attended_features, attended_shape, image_index, save_dir):
    attended_features = attended_features.view(attended_shape).cpu().detach().numpy()
    attended_features_mean = np.mean(attended_features, axis=0)

    mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
    attended_features_mean_resized = cv2.resize(attended_features_mean, (image.shape[2], image.shape[1]))
    threshold = np.percentile(attended_features_mean_resized, 65)
    mask[attended_features_mean_resized > threshold] = 1

    # Find the coordinates of the non-zero pixels in the mask
    keypoints = np.transpose(np.nonzero(mask))
    num_keypoints = keypoints.shape[0]

    # Consider every 4th pixel as a keypoint
    keypoints = keypoints[::4]

    # Create dummy descriptors
    descriptors = np.zeros((num_keypoints, 128))

    # Save the keypoints and descriptors
    np.save(os.path.join(save_dir, f'keypoints_{image_index}.npy'), keypoints)
    np.save(os.path.join(save_dir, f'descriptors_{image_index}.npy'), descriptors)

    # Iterate over keypoints and draw them on the image
    image_with_keypoints = np.transpose(image.cpu().numpy(), (1, 2, 0)).copy()
    image_with_keypoints = (image_with_keypoints * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    image_with_keypoints = np.clip(image_with_keypoints, 0, 255).astype(np.uint8)

    for y, x in keypoints:
        cv2.circle(image_with_keypoints, (x, y), 2, (255, 0, 0), -1)

    # Save the image with keypoints
    cv2.imwrite(os.path.join(save_dir, f'image_with_keypoints_{image_index}.png'), image_with_keypoints)


save_dir = "attention_overlays_masks"
os.makedirs(save_dir, exist_ok=True)
# Iterate over all images in the dataset
for i, (image1, image2, _) in enumerate(train_dataloader):
    print(i)
    image1, image2 = image1[0].to(device), image2[0].to(device)
    pose_params, attended_feat1, attended_feat2 = trained_model(image1.unsqueeze(0), image2.unsqueeze(0))
    save_attended_features_overlay(image1, attended_feat1.squeeze(), attended_shape, i * 2, save_dir)
    save_attended_features_overlay(image2, attended_feat2.squeeze(), attended_shape, i * 2 + 1, save_dir)