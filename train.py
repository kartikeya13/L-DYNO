import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
writer = SummaryWriter("Best_logs")

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

data_transforms = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = KITTIDataset('kitti/data', transform=data_transforms)
train_dataloader = DataLoader(train_dataset,shuffle=True, batch_size=32, num_workers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

def pose_loss_fn(predicted_pose, gt_pose):
    translation_loss = torch.norm(predicted_pose[:,:3] - gt_pose[:,:3],dim=1)
    rotation_loss = torch.norm(predicted_pose[:, 3:] - gt_pose[:, 3:],dim=1)
    
    loss = torch.mean( translation_loss + rotation_loss)
   # print("llllll",loss)
    return loss
    
def main():
    batch_size =32
    input_size = 256 * 18 * 18
    hidden_size = 512
    num_layers = 2
    output_size = 6
    attention_dim = 512
    model = SiameseNetwork(input_size, hidden_size, num_layers, output_size, attention_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 150
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_translation_loss = 0.0
        running_rotation_loss =0.0
        for i, data in enumerate(train_dataloader):
           # print("i:",i)
            img1, img2, gt_rel_pose = data
            img1, img2, gt_rel_pose = img1.to(device), img2.to(device), gt_rel_pose.to(device)
            optimizer.zero_grad()
            predicted_pose = model(img1, img2)
            translation_loss = torch.norm(predicted_pose[:,:3] - gt_rel_pose[:,:3],dim=1)
            rotation_loss = torch.norm(predicted_pose[:,3:] - gt_rel_pose[:,3:], dim=1)
            loss =pose_loss_fn(predicted_pose,gt_rel_pose)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
           # print("runn:",running_loss)
            running_translation_loss += translation_loss.mean().item()
           # print("hiii")
            running_rotation_loss += rotation_loss.mean().item()
        writer.add_scalar("Train Loss", running_loss / len(train_dataloader), epoch)
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)},Translation Loss: {running_translation_loss / len(train_dataloader)}, Rotation Loss: {running_rotation_loss /len(train_dataloader)}")
    writer.flush()
    torch.save(model.state_dict(), 'newcolor_att.pth')
if __name__ == "__main__":
    main()



