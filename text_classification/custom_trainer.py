import torch
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Ensure labels are on the correct device and of type long
        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.to(self.device).to(torch.long)

        # Get model outputs
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Ensure logits are on the correct device and float
        if logits is not None:
            logits = logits.to(self.device).float()

        # Create class weights tensor, using .clone().detach() to avoid the UserWarning
        class_weights_tensor = torch.tensor(self.class_weights, dtype=torch.float32).clone().detach().to(self.device)

        # Define loss function
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    
    def set_class_weights(self, class_weights):
        self.class_weights = class_weights
        
    def set_device(self, device):
        self.device = device