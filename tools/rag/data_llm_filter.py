"""
Usage Instructions:
    This script is designed to check whether txt files in a specified folder meet a given thematic requirement.
    Files that do not meet the requirement will be moved to an output folder.

    How to Use:
    python tools/rag/data_llm_filter.py \
        -i [input directory] -o [output directory] -t "topic" \
        -type [model]

    For our project：
    python tools/rag/data_llm_filter.py \
        -i docs/raw -o docs/filtered_out -t "emotional intelligence" \
        -type zhipuai


    Parameters:
        -i, --input_folder: Path to the input folder containing txt files.
        -o, --output_folder: Path to the output folder where non-compliant files will be moved.
        -t, --theme: The thematic content requirement (e.g., "toasting/alcohol culture/drinking/banquets").
        -d, --debug: Enables debug mode; prints the LLM output for each file.
        -m, --model: Specifies the LLM model to use (default: 'internlm/internlm2_5-7b-chat').
"""

import os
import argparse
from tqdm import tqdm
import shutil
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from zhipuai import ZhipuAI
from openai import OpenAI

load_dotenv()


class LLMProcessor:
    def __init__(
        self, model_type, model_name=None, api_key=None, cache_dir=None, device="cuda"
    ):
        self.model_type = model_type
        if model_type == "local":
            self.model, self.tokenizer = self.load_local_model(model_name, cache_dir)
            self.device = device
        elif model_type == "zhipuai":
            self.client = ZhipuAI(api_key=api_key)
        elif model_type == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE")
            )
            self.model_name = os.getenv("OPENAI_API_MODEL")

    def load_local_model(self, model_name, cache_dir):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            load_in_4bit=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        return model, tokenizer

    def process_message(self, system_message, user_message, debug=False):
        if self.model_type == "local":
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs.input_ids, max_length=12800, temperature=0.2
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            result = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
        elif self.model_type == "zhipuai":
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
            response = self.client.chat.completions.create(
                model="glm-4-air",
                messages=messages,
                temperature=0.1,  # 设置温度为0.1
            )
            result = response.choices[0].message.content
        elif self.model_type == "openai":
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
            )
            result = response.choices[0].message.content
        return result


def check_theme(processor, summary, theme, debug=False):
    system_message = "You are an inspection expert. Return true if the document contains any element related to the given topic. If it involves celebrities, specific advertisements, or duplicate content, just return false. Please return only true or false, and nothing else."
    user_message = f"Topic is:{theme}\n\nThe document need to be inspected is：{summary}"
    result = processor.process_message(system_message, user_message, debug=debug)

    if_true = "true" in result
    if debug:
        logger.info(f"Generated result: {result} , {if_true}")
    return if_true


def main():
    parser = argparse.ArgumentParser(description="Check if file match the topic")
    parser.add_argument(
        "-i", "--input_folder", type=str, required=True, help="input file directory"
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, required=True, help="output file directory"
    )
    parser.add_argument("-t", "--theme", type=str, required=True, help="topic required content")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug mode and print each LLM output result."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="internlm/internlm2_5-7b-chat",
        help="Specify the model type to use",
    )
    parser.add_argument(
        "-type",
        "--model_type",
        type=str,
        choices=["local", "zhipuai", "openai"],
        required=True,
        help="choose model type",
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    theme = args.theme
    debug = args.debug
    model_name = args.model
    model_type = args.model_type
    
    api_key = os.getenv("ZHIPUAI_API_KEY")
    error_log_path = os.path.join(output_folder, "error_log.txt")

    cache_dir = os.path.join(os.getenv("TIANJI_PATH", ""), "temp", "local_llm")
    processor = LLMProcessor(
        model_type, model_name=model_name, api_key=api_key, cache_dir=cache_dir
    )

    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(input_folder), desc="handle files"):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_folder, filename)
            try:
                with open(input_file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                is_relevant = check_theme(processor, content, theme, debug)
                if not is_relevant:
                    output_file_path = os.path.join(output_folder, filename)
                    shutil.move(input_file_path, output_file_path)
            except Exception as e:
                with open(error_log_path, "a", encoding="utf-8") as error_log:
                    error_log.write(f"handle filed: {filename}, error: {str(e)}\n")


if __name__ == "__main__":
    main()