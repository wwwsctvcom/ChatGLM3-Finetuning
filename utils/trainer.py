import torch
from tqdm import tqdm
from pathlib import Path
from utils.tools import get_lr


class Trainer:

    def __init__(self,
                 args=None,
                 model=None,
                 tokenizer=None,
                 optimizer=None,
                 scheduler=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            raise ValueError("tokenizer is None!")

        self.optimizer = optimizer
        if optimizer is None:
            raise ValueError("optimizer is None!")
        
        self.scheduler = scheduler

    def train(self, train_data_loader=None, test_data_loader=None):
        for epoch in range(1, self.args.epochs + 1):
            train_total_loss = 0
            self.model.train()
            with tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f'Epoch: {epoch}/{self.args.epochs}',
                      postfix=dict) as train_pbar:
                for step, batch in train_pbar:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # backward, calculate gradient
                    outputs = self.model(**batch, use_cache=False)
                    loss = outputs.loss

                    self.model.backward(loss)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.model.step()
                    self.model.zero_grad()

                    # lr scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()

                    train_total_loss += loss.item()

                    train_pbar.set_postfix(
                        **{'train average loss': train_total_loss / (step + 1), 'train loss': loss.item(),
                           'lr': get_lr(self.optimizer)})
            # test
            if test_data_loader is not None:
                test_total_loss = 0
                with tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc=f'Epoch: {epoch}/{self.args.epochs}',
                          postfix=dict) as test_pbar:
                    self.model.eval()
                    for step, batch in test_pbar:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch, use_cache=False)
                        loss = outputs.loss

                        # tqdm
                        test_pbar.set_postfix(
                            **{'test average loss': test_total_loss / (step + 1), 'test loss': loss.item()})
            self.save_model((Path(self.args.output_dir) / f"epoch_{epoch}").__str__())

    def save_lora_ckpt(self, lora_ckpt_path: str = None):
        unwrap_net = self.accelerator.unwrap_model(self.model)
        unwrap_net.save_pretrained(lora_ckpt_path)

    def load_ckpt(self, lora_ckpt_path: str = None):
        import os
        self.model.load_state_dict(
            torch.load(os.path.join(lora_ckpt_path, 'adapter_model.bin')), strict=False)

    def save_model(self, out_dir: str = None):
        if not Path(out_dir).exists():
            Path(out_dir).mkdir()
        self.model.save_pretrained(out_dir, torch_dtype=torch.float16)
        self.tokenizer.save_pretrained(out_dir)