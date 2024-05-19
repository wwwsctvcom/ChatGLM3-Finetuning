import json
import torch
from torch.utils.data import Dataset


class GLM3Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        self.all_data = []
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                skip_flag = False

                src_tokens = [tokenizer.get_command("<|user|>")] + tokenizer.encode("\n", add_special_tokens=False) + \
                             tokenizer.encode(sample["instruction"] + sample["input"], add_special_tokens=False)

                if len(src_tokens) > max_src_len:
                    # 当输入内容超长时，随向后截断
                    src_tokens = src_tokens[:max_src_len]
                    skip_flag = True

                max_tgt_len = max_len - 6 - len(src_tokens)
                tgt_tokens = [tokenizer.get_command("<|assistant|>")] + tokenizer.encode("\n", add_special_tokens=False) + \
                             tokenizer.encode(sample["output"], add_special_tokens=False)

                if len(tgt_tokens) > max_tgt_len:
                    # 当输出内容超长时，随向后截断
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                    skip_flag = True

                # ChatGLM3需要增加[gMASK]、sop两个标记
                input_ids = [tokenizer.get_command("[gMASK]"),
                             tokenizer.get_command("sop")] + src_tokens + tgt_tokens + [tokenizer.eos_token_id]
                context_length = len(src_tokens) + 2
                labels = [-100] * context_length + input_ids[context_length:]

                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids": input_ids, "labels": labels})
        print("the number of skipping data is {}".format(skip_data_number))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        lengths = [len(instance["input_ids"]) for instance in batch]
        batch_max_len = max(lengths)

        input_ids_batch, labels_batch = [], []
        for instance in batch:
            input_ids = instance["input_ids"]
            labels = instance["labels"]

            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)

        return {"input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "labels": torch.tensor(labels_batch, dtype=torch.long)}


# if __name__ == "__main__":
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="/root/autodl-tmp/ChatGLM3-Finetuning/cache/ZhipuAI/chatglm3-6b", trust_remote_code=True)
#     print(tokenizer.pad_token_id)

#     train_dataset = GLM3Dataset(data_path="/root/autodl-tmp/ChatGLM3-Finetuning/data/train_data.jsonl",
#                                 tokenizer=tokenizer,
#                                 max_len=1024,
#                                 max_src_len=256,
#                                 is_skip=True)

#     from torch.utils.data import DataLoader
#     data_collactor = DataCollator(tokenizer)
#     train_dataloader = DataLoader(train_dataset,
#                                   collate_fn=data_collactor,
#                                   batch_size=2,
#                                   num_workers=2,
#                                   shuffle=False)

#     for batch in train_dataloader:
#         print(batch)
#         break