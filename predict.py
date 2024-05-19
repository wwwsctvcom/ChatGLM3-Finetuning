import argparse
from loguru import logger
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")


def set_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--model_name_or_path", 
                        default="./cache/ZhipuAI/chatglm3-6b", 
                        type=str)
    parser.add_argument("--lora_name_or_path", default="/root/autodl-tmp/ChatGLM3-Finetuning/output/epoch_2", type=str)

    # param
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def predict(args, model, tokenizer, query):
    response, _ = model.chat(tokenizer, 
                             query, 
                             max_length=args.max_length, 
                             num_beams=args.num_beams, 
                             do_sample=args.do_sample, 
                             top_p=args.top_p, 
                             temperature=args.temperature)
    return response


if __name__ == "__main__":
    args = set_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path,
                                      load_in_8bit=False,
                                      trust_remote_code=True,
                                      device_map='auto')
    if args.lora_name_or_path:
        logger.info(f"Loading lora model {args.lora_name_or_path}")
        model = PeftModel.from_pretrained(model, args.lora_name_or_path)
        model = model.merge_and_unload()  # merge lora

    answer = predict(args, model, tokenizer, '文本分类任务：将一段用户给手机的评论进行分类。下面是一些范例：第一天用出现两次卡顿，对国产中低端机器始终没有信任可言')
    print(answer)
