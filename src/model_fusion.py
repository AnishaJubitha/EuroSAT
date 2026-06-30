import torch.nn as nn
from src.utils import *
from src.models import *

class FusionResNet18(nn.Module):
    def __init__(self, rgb_checkpoint=None, ms_checkpoint=None, freeze=False):

            super().__init__()

            device = get_device()

            self.rgb_branch = resnet18_rgb_model(False)
            self.ms_branch = resnet18_ms_model(False)

            if rgb_checkpoint is not None:
                  rgb_path = CHECKPOINT_PATH / rgb_checkpoint

                  self.rgb_branch.load_state_dict(
                        torch.load(rgb_path, map_location=device)
                  )

                  print(f"Loaded RGB Model Checkpoint.")

            if ms_checkpoint is not None:
                  ms_path = CHECKPOINT_PATH / ms_checkpoint

                  self.ms_branch.load_state_dict(
                        torch.load(ms_path, map_location=device)
                  )

                  print('Loaded MS Model Checkpoint.')

            
            # Changing the Final Classifier - getting feature dimensions

            rgb_feature_dim = self.rgb_branch.fc.in_features
            ms_feature_dim = self.ms_branch.fc.in_features


            # Removing final classification 

            self.rgb_branch.fc = nn.Identity()
            self.ms_branch.fc = nn.Identity()

            if freeze:
                for param in self.rgb_branch.parameters():
                      param.requires_grad = False

                for param in self.ms_branch.parameters():
                      param.requires_grad = False

                    
            # Fusion Classifier

            self.classifier = nn.Sequential(
                  nn.Linear(rgb_feature_dim + ms_feature_dim, 512),
                  nn.ReLU(), 
                  nn.Dropout(p=0.3),
                  nn.Linear(512, NUM_CLASSES)
            )

    def forward(self, rgb_image, ms_image):
          
        rgb_features = self.rgb_branch(rgb_image)
        ms_features = self.ms_branch(ms_image)

        fused_features = torch.cat(
              [rgb_features, ms_features],
              dim=1
          )
        
        output = self.classifier(fused_features)

        return output
    