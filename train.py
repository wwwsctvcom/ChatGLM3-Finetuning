import argparse
import deepspeed
import json
import math
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from utils.tools import *
from utils.trainer import Trainer
from utils.dataset import GLM3Dataset, DataCollator



def set_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model_name_or_path", default="", type=str, required=True)

    # dataset
    parser.add_argument("--train_path", default="data/train_data.jsonl", type=str, help="train dataset !")
    parser.add_argument("--test_path", default="data/test_data.jsonl", type=str, help="test dataset !")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--max_src_len", type=int, default=256)
    parser.add_argument("--is_skip", action='store_true', help="skip the too long data!")

    # train
    parser.add_argument("--train_type", type=str, default="lora", choices=["lora", "all"],
                        help="train type for lora or all parameters training !")
    parser.add_argument("--output_dir", type=str, default="output", help="")
    parser.add_argument("--epochs", type=int, default=1, help="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="batch size for train and test !")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers for dataloader !")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for training!")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="")
    
    # deepspeed
    parser.add_argument("--ds_file", type=str, default="ds_zero2_no_offload.json")
    parser = deepspeed.add_config_arguments(parser)

    # lora
    parser.add_argument("--lora_dim", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_module_name", type=str, default="query_key_value")
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()

    # seed
    seed_everything(args.seed)

    # distributed
    if args.local_rank == -1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        logger.info("distributing training !")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()
    logger.info(f"Global rank: {args.global_rank} !")

    # distributed training process synchronize
    torch.distributed.barrier()

    # load tokenizer and config
    logger.info(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True, device_map='auto')

    # load model not in 8bit
    model = AutoModel.from_pretrained(args.model_name_or_path,
                                      load_in_8bit=False,
                                      trust_remote_code=True,
                                      device_map='auto')

    # gradient_checkpointing for saving cuda resource
    if args.gradient_checkpointing:
        model.supports_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # close model cache to ignore some warning, but need to reopen when inference.
    model.config.use_cache = False

    if args.train_type == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False,
            r=args.lora_dim,
            target_modules=args.lora_module_name.split(","),
            bias="none",
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
    
    model.is_parallelizable = True
    model.model_parallel = True
    model.print_trainable_parameters()

    # train dataset
    data_collator = DataCollator(tokenizer)
    train_dataset = GLM3Dataset(data_path=args.train_path,
                                tokenizer=tokenizer,
                                max_len=args.max_len,
                                max_src_len=args.max_src_len,
                                is_skip=args.is_skip)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  batch_size=args.per_device_train_batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True)

    # test dataset
    test_dataset = GLM3Dataset(data_path=args.test_path,
                               tokenizer=tokenizer,
                               max_len=args.max_len,
                               max_src_len=args.max_src_len,
                               is_skip=args.is_skip)
    test_dataloader = DataLoader(test_dataset,
                                 collate_fn=data_collator,
                                 batch_size=args.per_device_train_batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False)

    # deepspeed
    with open(args.ds_file, "r", encoding="utf-8") as ds_reader:
        ds_config = json.load(ds_reader)
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps

    ds_config["optimizer"]["params"]["lr"] = args.learning_rate
    ds_config["optimizer"]["params"]["betas"] = (0.9, 0.95)
    ds_config["optimizer"]["params"]["eps"] = 1e-8
    ds_config["optimizer"]["params"]["weight_decay"] = 0.1
    num_training_steps = args.epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    ds_config["scheduler"]["params"]["total_num_steps"] = num_training_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.learning_rate
    ds_config["scheduler"]["params"]["warmup_min_lr"] = args.learning_rate * 0.1

    # init deepspeed
    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, args=args, config=ds_config,
                                                             dist_init_required=True)
    model.train()

    # start train
    trainer = Trainer(args=args, 
                      model=model,
                      tokenizer=tokenizer,
                      optimizer=optimizer,
                      scheduler=lr_scheduler)
    
    trainer.train(train_data_loader=train_dataloader,
                  test_data_loader=test_dataloader)