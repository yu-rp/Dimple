import os

os.environ["WANDB_PROJECT"] = "dimple"
import torch
import json, warnings
import math
import numpy as np
from accelerate import Accelerator
import logging 
from transformers.utils import logging as hf_logging
from utils.logger_utils import setup_logging_for_main_process, add_file_handler

accelerator = Accelerator()
setup_logging_for_main_process()

logger = hf_logging.get_logger("dimple")
logger.info("Logging started")

from DLM_trainer import DLMTrainer, AutoRegressiveTrainer
from transformers import TrainingArguments, AutoProcessor, Trainer
from datasets import Dataset

from accelerate import PartialState
device_string = PartialState().process_index

from datasets import Dataset, load_dataset

from models.processing_dimple import DimpleProcessor
from models.modeling_dimple import DimpleModel
from models.tokenization_dimple import DimpleTokenizer
from models.image_processing_dimple import DimpleImageProcessor

from utils.dataset import CustomizedDataProcessor, CustomizedDataCollator, load_llava_next_dataset, load_llava_alignment_dataset
from utils.dlm_args import DLMTrainingArguments
from utils import utils

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Train Multimodal-DLLM ")
    parser.add_argument("--seed", type=int, default=42)

    # wandb args
    parser.add_argument("--resume-wandb", action='store_true', default=False)
    parser.add_argument("--wandb-run-id", type=str, default='')

    # model args
    parser.add_argument("--model", type=str, default = "", help="Model name")
    parser.add_argument("--ckpt", type=str, default = "", help="CKPT")
    parser.add_argument("--causal_attention", type=str, default = "no", help="whether to use causal attention or not")
    parser.add_argument("--max-pixel-scale", type=float, default = 24, help="24 for max-pixel-value 336, 37 for max-pixel-value 518, -1 for original max-pixel-value")
    
    # dataset args
    parser.add_argument("--dataset", type=str, default="llava_next", help="Dataset name")
    parser.add_argument("--shuffle-training-data", type=str, default="yes", help="whether shuffle the training data")
    parser.add_argument("--ans-seq-length", type=int, default=64, help="Sequence length of each answer, if fixed padding strategy is used")
    parser.add_argument("--answer-padding-strategy", type=str, default="dynamic", help="Which padding strategy to use for the answer. fixed or dynamic")
    parser.add_argument("--only-text", action='store_true', default=False, help="Only use text data")
    parser.add_argument("--use-shard", action='store_true', default=False, help="Whether to use shard dataset to speed up the training")
    
    # training hyperparameter
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--wd", type=float, default=0.00, help="Weight decay")
    parser.add_argument("--warm_up", type=float, default=0.03, help="Warm up ratio")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--epoch", type=int, default=2, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=9000, help="Max steps")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16) 
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--transition_mode", type=str, default="linear", help="Transition mode: linear, follow-token-order")
    parser.add_argument("--mask_strategy", type=str, default="single", help="Mask strategy: dual, single")
    parser.add_argument("--rescale_loss", type=str, default="yes", help="whether to rescale the loss or not")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")

    # training: for saving, debugging, and others
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--trainer-type", type=str, default="dlm", help="dlm or autoregressive")
    parser.add_argument("--output-dir", type=str, default="output", help="Path to the output directory")
    parser.add_argument("--save-steps", type=int, default=400, help="Save steps")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Eval steps")
    parser.add_argument("--stage", type=str, default="instruct", help="Training stage")
    parser.add_argument("--mark", type=str, default="", help="Additional mark for the output folder")
    parser.add_argument("--save-ckpt", action='store_true', default=False, help="whether to save the checkpoint or not")
    parser.add_argument("--not-save-final", action='store_true', default=False, help="whether to save the final model or not")
    parser.add_argument("--close-report", action='store_true', default=False, help="whether to close the report or not")

    return parser.parse_args()

if __name__ == "__main__":
    args = argparse()

    # get num of GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {num_gpus}")

    real_batch_size = args.per_device_train_batch_size * num_gpus * args.gradient_accumulation_steps
    if args.mask_strategy == "dual":
        logger.info("Dual mask strategy is used. The real batch size x2.")
        logger.info("Dual mask strategy is used. The real batch size per gpu x2. But the args.per_device_train_batch_size should not be changed")
        real_batch_size *= 2
    logger.info(f"Real batch size: {real_batch_size}. Provided batch size: {args.batch_size}" )
    if real_batch_size != args.batch_size:
        logger.info("The real batch size is not equal to the provided batch size. The batch size arg is overwritten.")
        args.batch_size = real_batch_size

    HASH_STR = f"_asl_{args.ans_seq_length}_aps_{args.answer_padding_strategy}_msl_{args.max_seq_length}_mps_{args.max_pixel_scale}"
    HASH_STR += f"_shuffle_{args.shuffle_training_data}_causal_attn_{args.causal_attention}"
    HASH_STR += f"_loss_rescale_{args.rescale_loss}_onlytxt_{args.only_text}_max_grad_norm_{args.max_grad_norm}"
    HASH_STR += f"_transition_mode_{args.transition_mode}_mask_strategy_{args.mask_strategy}"
    OUTPUT_FOLDER = f"{args.model}_{args.dataset}_{args.stage}_lr_{args.learning_rate}_wd_{args.wd}_bs_{args.batch_size}_tt_{args.trainer_type}"
    OUTPUT_FOLDER += f"_{utils.stable_hash(HASH_STR)}_seed_{args.seed}"
    if args.mark:
        OUTPUT_FOLDER = f"{OUTPUT_FOLDER}_{args.mark}"
    OUTPUT_FOLDER = os.path.join(args.output_dir, OUTPUT_FOLDER)
    args.output_folder = OUTPUT_FOLDER
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    add_file_handler(logger, OUTPUT_FOLDER, filename="training.log")
    logger.info(f"Output folder: {OUTPUT_FOLDER}")

    if args.resume_wandb and args.wandb_run_id:
        os.environ["WANDB_RESUME"] = "must"
        os.environ["WANDB_RUN_ID"] = args.wandb_run_id

    if args.shuffle_training_data == "yes":
        args.shuffle_training_data = True
        logger.info("Shuffle training data")
    elif args.shuffle_training_data == "no":
        args.shuffle_training_data = False
        logger.info("Do not shuffle training data")
    else:
        raise ValueError("shuffle_training_data should be yes or no")

    if args.causal_attention == "yes":
        args.causal_attention = True
        logger.info("Use causal attention")
    elif args.causal_attention == "no":
        args.causal_attention = False
        logger.info("Not use causal attention")
    else:
        raise ValueError("causal_attention should be yes or no")

    if args.rescale_loss == "yes":
        args.rescale_loss = True
        logger.info("rescale loss")
    elif args.rescale_loss == "no":
        args.rescale_loss = False
        logger.info("Not rescale loss")
    else:
        raise ValueError("rescale_loss should be yes or no")

    if args.trainer_type == "dlm":
        pass
    elif args.trainer_type == "autoregressive":
        assert args.causal_attention, "Should use causal attention, when using autoregressive trainer"
        assert args.mask_strategy == "single", "mask strategy does not function"
    else:
        raise ValueError("Trainer type should be dlm or autoregressive")

    if not args.save_ckpt:
        args.save_steps = args.max_steps * 10

    logger.info("Hyper-parameters:")
    logger.info(json.dumps(vars(args), indent = 2))

    if args.model == 'Dimple-ar-instruct':
        model = DimpleModel.from_pretrained("/folder/to/the/model/after/ar-instruction-tuning", torch_dtype=torch.bfloat16)
        if args.causal_attention:
            model.config.full_attn_mask = False
        else:
            model.config.full_attn_mask = True
        tokenizer = DimpleTokenizer.from_pretrained("/folder/to/the/model/after/ar-instruction-tuning", padding_side="right")
        img_processor = DimpleImageProcessor.from_pretrained("/folder/to/the/model/after/ar-instruction-tuning")
        if args.max_pixel_scale > 0:
            img_processor.max_pixels = (14*args.max_pixel_scale) ** 2
            img_processor.size["max_pixels"] = (14*args.max_pixel_scale) ** 2
            logger.info(f"Using max pixel value: {img_processor.max_pixels}={math.sqrt(img_processor.max_pixels)}")
        else:
            logger.info("Using original max pixel value")
        processor = DimpleProcessor(
            image_processor=img_processor, 
            tokenizer=tokenizer, 
            chat_template=tokenizer.chat_template,
        )
    elif args.model == 'Dimple-align':
        model = DimpleModel.from_pretrained("/folder/to/the/model/after/alignment", torch_dtype=torch.bfloat16)
        if args.causal_attention:
            model.config.full_attn_mask = False
        else:
            model.config.full_attn_mask = True
        tokenizer = DimpleTokenizer.from_pretrained("/folder/to/the/model/after/alignment", padding_side="right")
        img_processor = DimpleImageProcessor.from_pretrained("/folder/to/the/model/after/alignment")
        if args.max_pixel_scale > 0:
            img_processor.max_pixels = (14*args.max_pixel_scale) ** 2
            img_processor.size["max_pixels"] = (14*args.max_pixel_scale) ** 2
            logger.info(f"Using max pixel value: {img_processor.max_pixels}={math.sqrt(img_processor.max_pixels)}")
        else:
            logger.info("Using original max pixel value")
        processor = DimpleProcessor(
            image_processor=img_processor, 
            tokenizer=tokenizer, 
            chat_template=tokenizer.chat_template,
        )
    elif args.model == 'Dimple-raw':
        model = DimpleModel.from_pretrained("/folder/to/the/raw/model", torch_dtype=torch.bfloat16)
        if args.causal_attention:
            model.config.full_attn_mask = False
        else:
            model.config.full_attn_mask = True
        tokenizer = DimpleTokenizer.from_pretrained("/folder/to/the/raw/model", padding_side="right")
        img_processor = DimpleImageProcessor.from_pretrained("/folder/to/the/raw/model")
        if args.max_pixel_scale > 0:
            img_processor.max_pixels = (14*args.max_pixel_scale) ** 2
            img_processor.size["max_pixels"] = (14*args.max_pixel_scale) ** 2
            logger.info(f"Using max pixel value: {img_processor.max_pixels}={math.sqrt(img_processor.max_pixels)}")
        else:
            logger.info("Using original max pixel value")
        processor = DimpleProcessor(
            image_processor=img_processor, 
            tokenizer=tokenizer, 
            chat_template=tokenizer.chat_template,
        )
    else:
        raise NotImplementedError("Model not implemented")

    # calcualte the parameter size for each submodule
    logger.info("Model size: {} M".format(sum([np.prod(p.size()) for p in model.parameters()])/1024/1024))
    logger.info("Image Encoder size: {} M".format(sum([np.prod(p.size()) for p in model.visual.parameters()])/1024/1024))
    logger.info("LM size: {} M".format(sum([np.prod(p.size()) for p in model.model.parameters()])/1024/1024))
    logger.info("LM Head Size: {} M".format(sum([np.prod(p.size()) for p in model.lm_head.parameters()])/1024/1024))
    logger.info("Projector size: {} M".format(sum([np.prod(p.size()) for p in model.img_projector.parameters()])/1024/1024))

    if args.stage == "instruct":
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.model.parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
        for param in model.img_projector.parameters():
            param.requires_grad = True
    elif args.stage == "diffusion-recovery":
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.model.parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
        for param in model.img_projector.parameters():
            param.requires_grad = False
    elif args.stage == "align":
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        for param in model.img_projector.parameters():
            param.requires_grad = True
    else:
        raise NotImplementedError("Stage not implemented")

    logger.info("Model Trainable Params: {} M".format(sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])/1024/1024))
    logger.info("Model Non-Trainable Params: {} M".format(sum([np.prod(p.size()) for p in model.parameters() if not p.requires_grad])/1024/1024))

    logger.info("Image Encoder Trainable Params: {} M".format(sum([np.prod(p.size()) for p in model.visual.parameters() if p.requires_grad])/1024/1024))
    logger.info("Image Encoder Non-Trainable Params: {} M".format(sum([np.prod(p.size()) for p in model.visual.parameters() if not p.requires_grad])/1024/1024))

    logger.info("LM Trainable Params: {} M".format(sum([np.prod(p.size()) for p in model.model.parameters() if p.requires_grad])/1024/1024))
    logger.info("LM Non-Trainable Params: {} M".format(sum([np.prod(p.size()) for p in model.model.parameters() if not p.requires_grad])/1024/1024))

    logger.info("LM Head Trainable Params: {} M".format(sum([np.prod(p.size()) for p in model.lm_head.parameters() if p.requires_grad])/1024/1024))
    logger.info("LM Head Non-Trainable Params: {} M".format(sum([np.prod(p.size()) for p in model.lm_head.parameters() if not p.requires_grad])/1024/1024))

    logger.info("Projector Trainable Params: {} M".format(sum([np.prod(p.size()) for p in model.img_projector.parameters() if p.requires_grad])/1024/1024))
    logger.info("Projector Non-Trainable Params: {} M".format(sum([np.prod(p.size()) for p in model.img_projector.parameters() if not p.requires_grad])/1024/1024))
    
    # load dataset
    if args.dataset == 'llava_alignment':
        logger.info("Loading llava_alignment dataset. Not actually use the eval dataset.")
        train_dataset = load_llava_alignment_dataset( split = "train", seed = args.seed, shuffle = args.shuffle_training_data, shard = args.use_shard)
    elif args.dataset == 'llava_next':
        logger.info("Loading llava_next dataset. Not actually use the eval dataset.")
        train_dataset = load_llava_next_dataset(seed = args.seed, shuffle = args.shuffle_training_data, shard = args.use_shard)
    else:
        raise NotImplementedError("Dataset not implemented")

    data_processor = CustomizedDataProcessor(
        tokenizer=tokenizer, 
        image_processor=img_processor, 
        ans_seq_length = args.ans_seq_length,
        answer_padding_strategy = args.answer_padding_strategy,
        max_seq_length = args.max_seq_length,
        only_text = args.only_text,
        )

    # The processed dataset will contain the following keys:
    # - images: list of image file paths
    # - input_ids: list of unshifted token ids, length = #tokens
    # - labels: list of unshifted labels for the input ids, length = #tokens
    # - mask_locations: list of unshifted boolean values indicating the locations can be masked, length = #tokens
    train_processed_dataset = train_dataset.map(
        lambda item: data_processor.process(item),
        batched=False,
        remove_columns=train_dataset.column_names,
    )
    
    data_collator = CustomizedDataCollator(
        tokenizer=tokenizer, 
        image_processor=img_processor,
        position_ids_function=model.get_rope_index,
    )

    training_args = TrainingArguments(
        resume_from_checkpoint = args.ckpt if args.ckpt else None,
        learning_rate=args.learning_rate,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        max_grad_norm = args.max_grad_norm,
        weight_decay = args.wd,
        warmup_ratio = float(args.warm_up) if args.warm_up < 1 else 0.0,
        warmup_steps = int(args.warm_up) if args.warm_up > 1 else 0,
        num_train_epochs= args.epoch,
        max_steps = args.max_steps,
        logging_steps = 1,
        save_strategy="steps",
        save_steps = args.save_steps,
        save_total_limit = 20,
        output_dir=OUTPUT_FOLDER,
        optim = "adamw_torch",
        seed = args.seed,
        bf16 = True,
        report_to = 'wandb' if not args.close_report else 'none',
        accelerator_config = {'dispatch_batches': False},
        ddp_find_unused_parameters = False,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        batch_eval_metrics = True,
        remove_unused_columns=False,
        dataloader_prefetch_factor = 32 if args.use_shard else None,
        dataloader_num_workers = 64 if args.use_shard else 0,
    )

    if args.trainer_type == "autoregressive":

        trainer = AutoRegressiveTrainer(
            loss_rescale=1/args.gradient_accumulation_steps if args.rescale_loss else 1,
            model=model,
            args=training_args,
            train_dataset=train_processed_dataset,
            data_collator=data_collator,
        )

    elif args.trainer_type == "dlm":
        dlm_args = DLMTrainingArguments(
            mask_token_id=tokenizer.mask_token_id,
            transition_mode=args.transition_mode,
            seq_length=args.max_seq_length,
            loss_rescale=1/args.gradient_accumulation_steps if args.rescale_loss else 1,
            mask_strategy=args.mask_strategy,
            linear_clamp_shift = args.linear_clamp_shift,
            linear_clamp_max = args.linear_clamp_max,
            linear_clamp_min = args.linear_clamp_min,
        )

        trainer = DLMTrainer(
            model=model,
            dlm_args=dlm_args,
            args=training_args,
            train_dataset=train_processed_dataset,
            data_collator=data_collator,
        )
    else:
        raise NotImplementedError("Trainer type not implemented")

    if not args.debug:
        if args.ckpt:
            trainer.train(
                resume_from_checkpoint=args.ckpt
            )
        else:
            trainer.train()
        if args.not_save_final:
            pass
        else:
            model.save_pretrained(OUTPUT_FOLDER)
            processor.save_pretrained(OUTPUT_FOLDER)