import os
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(os.listdir(os.path.join(root_dir, 'image_00')))
        self.gt_poses = np.loadtxt('00.txt')

    def __len__(self):
        return len(self.image_paths) - 1

    def __getitem__(self, idx):
        img1 = Image.open(os.path.join(self.root_dir, 'image_00', self.image_paths[idx]))
        img2 = Image.open(os.path.join(self.root_dir, 'image_00', self.image_paths[idx + 1]))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        gt_rel_pose = self.gt_poses[idx]
        
       # pose1 = np.vstack((pose1, np.array([0, 0, 0, 1])))
       # pose2 = np.vstack((pose2, np.array([0, 0, 0, 1])))
       # gt_rel_pose = np.linalg.inv(pose1).dot(pose2)
        return img1, img2, torch.from_numpy(gt_rel_pose).float()
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        attention_weights = self.attention_network(features)
        attended_features = features * attention_weights
        return attended_features
        
        
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

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,attention_dim):
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
        self.attention_layer = AttentionLayer(input_size, attention_dim) # Add attention layer
        self.pose_estimation = PoseEstimationLSTM(input_size, hidden_size, num_layers, output_size)

    def forward(self, img1, img2):
        feat1, feat2 = self.feature_extractor(img1), self.feature_extractor(img2)
        feat1_flat = feat1.view(feat1.size(0), -1)
        feat2_flat = feat2.view(feat2.size(0), -1)
        attended_feat1 = self.attention_layer(feat1_flat) # Apply attention to features
        attended_feat2 = self.attention_layer(feat2_flat) # Apply attention to features
        feats = torch.stack([attended_feat1, attended_feat2], dim=1)
        pose_params = self.pose_estimation(feats)
        return pose_params






batch_size =32
input_size = 256 * 18 * 18
hidden_size = 512
num_layers = 2
output_size = 6
attention_dim = 512
# Load the trained model
model = SiameseNetwork(input_size, hidden_size, num_layers, output_size, attention_dim).to(device)
model.load_state_dict(torch.load('newcolor_att.pth'))
model.eval()

# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define test dataset
test_dataset = KITTIDataset('kitti/data', transform=data_transforms)

# Loop over test dataset and save output in a txt file
with open('output_trans.txt', 'w') as f:
    for i in range(len(test_dataset)):
        img1, img2, gt_rel_pose = test_dataset[i]
        img1, img2, gt_rel_pose = img1.to(device), img2.to(device), gt_rel_pose.to(device)
        predicted_pose = model(img1.unsqueeze(0), img2.unsqueeze(0)).detach().cpu().numpy()[0]
        print(predicted_pose.reshape(1,-1))
        np.savetxt(f, predicted_pose.reshape(1, -1))

