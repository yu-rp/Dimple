import torch
import torch.nn as nn
from transformers import Trainer
from transformers.utils import logging

logger = logging.get_logger("dimple."+__name__)

class AutoRegressiveTrainer(Trainer):
    def __init__(self, loss_rescale, **kwargs):
        logger.debug("AutoRegressiveTrainer is initialized")
        self.loss_rescale = loss_rescale

        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            loss, _ = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        else:
            loss = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        loss = loss * self.loss_rescale
        
        return (loss, None) if return_outputs else loss

class DLMTrainer(Trainer):
    def __init__(self, dlm_args, **kwargs):
        self.sampling_eps = 1e-3
        self.mask_token_id = dlm_args.mask_token_id
        self.loss_fn = nn.CrossEntropyLoss(reduction='none') #CrossEntropyLoss(inplace_backward=True,reduction='none')

        self.dlm_args = dlm_args

        super().__init__(**kwargs)

    def transition(self, inputs, sigma, mask_token_id):
        x_0 = inputs['input_ids']
        maskable_mask = inputs['mask_locations']

        move_chance = sigma
        move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
        x_t = torch.where(move_indices, mask_token_id, x_0)

        inputs["input_ids"] = x_t
        return inputs, sigma

    def repeat_tensor_independent(self, tensor, repeats, dim=0):
        copies = [tensor.detach().clone() for _ in range(repeats)]
        return torch.cat(copies, dim=dim)

    def transition_dual(self, inputs, sigma, mask_token_id):

        x_0 = inputs['input_ids']
        maskable_mask = inputs['mask_locations']

        move_chance = sigma
        random_mask = torch.rand(*x_0.shape, device=x_0.device) < move_chance
        move_indices_prime = (random_mask) & maskable_mask
        move_indices_dual = (~random_mask) & maskable_mask
        move_indices = torch.cat([move_indices_prime, move_indices_dual], dim=0)

        if inputs.get("pixel_values", None) is not None:
            inputs["pixel_values"] = self.repeat_tensor_independent(inputs["pixel_values"], repeats = 2, dim=0)
            inputs["image_grid_thw"] = self.repeat_tensor_independent(inputs["image_grid_thw"], repeats = 2, dim=0)
        inputs["input_ids"] = self.repeat_tensor_independent(inputs["input_ids"], repeats = 2, dim=0)
        inputs["attention_mask"] = self.repeat_tensor_independent(inputs["attention_mask"], repeats = 2, dim=0)
        inputs["labels"] = self.repeat_tensor_independent(inputs["labels"], repeats = 2, dim=0)
        inputs["mask_locations"] = self.repeat_tensor_independent(inputs["mask_locations"], repeats = 2, dim=0)
        inputs["position_ids"] = self.repeat_tensor_independent(inputs["position_ids"], repeats = 2, dim=1)
        inputs["rope_deltas"] = self.repeat_tensor_independent(inputs["rope_deltas"], repeats = 2, dim=0)
        if inputs.get("order_token", None) is not None:
            inputs["order_token"] = self.repeat_tensor_independent(inputs["order_token"], repeats = 2, dim=0)

        inputs["input_ids"] = torch.where(move_indices, mask_token_id, inputs["input_ids"])

        return inputs, self.repeat_tensor_independent(sigma, repeats = 2, dim=0) 

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        input_ids = inputs['input_ids']
        t = (1 - self.sampling_eps) * torch.rand(input_ids.shape[0], device=input_ids.device) + self.sampling_eps

        if self.dlm_args.transition_mode == "linear":
            logger.warning_once(f"linear modification is triggered.")

            if self.dlm_args.mask_strategy == "single":
                sigma = t
                dsigma = 1

                inputs, sigma = self.transition(
                    inputs, 
                    sigma[:, None], 
                    mask_token_id=self.mask_token_id)
            elif self.dlm_args.mask_strategy == "dual":

                sigma = t
                dsigma = 1

                inputs, sigma = self.transition_dual(
                    inputs, 
                    sigma[:, None],
                    mask_token_id=self.mask_token_id
                    )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        input_ids = inputs['input_ids']
        
        masked_ids = input_ids == self.mask_token_id # 1 for masked token, 0 for unmasked token
        
        logits = model(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'],
            position_ids=inputs['position_ids'],
            pixel_values=inputs.get('pixel_values',None),
            image_grid_thw=inputs.get('image_grid_thw',None),
            rope_deltas=inputs['rope_deltas'],
        ).logits

        logits = logits[:,:-1]

        labels = inputs['labels']
        labels = labels.masked_fill(~masked_ids, -100)
        labels = labels[:,1:]

        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        ).view(logits.shape[0], logits.shape[1])

        logger.warning_once(f"loss is calculated with dsigma, {type(dsigma)}, {dsigma}")
        loss = dsigma * loss
        loss = loss[labels != -100]  # remove padding
        loss = loss.mean()  # mean loss
        loss = loss * self.dlm_args.loss_rescale

        return (loss, None) if return_outputs else loss