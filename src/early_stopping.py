from src.config import *
import torch

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, checkpoint_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = 'max'

        if checkpoint_path is not None:
            self.checkpoint_path = CHECKPOINT_PATH / checkpoint_path
        else:
            self.checkpoint_path = None

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def is_improvement(self, score):
        if self.best_score is None:
            return True
        
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        
        raise ValueError('mode must be either min or max.')
    
    def __call__(self, score, model):
        if self.is_improvement(score):
            self.best_score = score
            self.counter = 0

            if self.checkpoint_path is not None:
                torch.save(model.state_dict(), self.checkpoint_path)

            return True
        
        self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return False