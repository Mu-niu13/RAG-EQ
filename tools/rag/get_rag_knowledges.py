"""
Usage Instructions:
    This script is used to extract information from text files in a specified folder and generate knowledge corpus in JSON format.

    How to use:
    python get_rag_knowledges.py -f [input directory] -o <output directory> -m <model type> [-d]

    For our project:
        python tools/rag/get_rag_knowledges.py \
            -f docs/raw -o docs/clean/knowledges.json -m zhipu


    Parameters:
        - `-f`, `--folder_path`: Specifies the path to the input folder containing the text files.  
        - `-o`, `--output_file`: Specifies the path for the generated JSON output file.  
        - `-m`, `--model`: Specifies the model type to use. Options are `"zhipu"`, `"deepseek"`, and `"local"`. Default is `"zhipu"`.  
        - `-d`, `--debug`: Enables debug mode to print the LLM output for each entry.

    Sample JSON Output Format:

        ```json
        {
            "Section Title": "Summary of the section 1. 2. 3."
        }
        {
            "Section Title": "Summary of the section 1. 2. 3."
        }
        ```
"""

import os
import json
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from zhipuai import ZhipuAI
from openai import OpenAI

load_dotenv()

SUMMARY_PROMPT = """
You are an expert in preparing conversational knowledge base materials. Your task is to transform the content I provide into declarative knowledge statements for use in a knowledge retrieval system. When summarizing, follow these guidelines:

- Respond entirely in English.  
- If the content contains step-by-step methods, combine them into a single cohesive summary.  
- If names of people or authors are mentioned, ignore them or replace them with generic references.  
- The final summary should be comprehensive and structured as continuous knowledge clauses, similar to entries in a reference system. Do not use numbered lists like 1, 2, 3, etc. — just continuous, well-organized knowledge segments.  

Return only the summarized knowledge content. The original text to be summarized is as follows:
"""

TITLE_PROMPT = """
Please summarize the following content with a concise title of no more than 20 words, focusing only on the subject matter and excluding any personal names.
"""


def get_summary(content, model_type="zhipu", debug=False):
    return get_llm_response(SUMMARY_PROMPT + content, model_type, debug)


def get_title(content, model_type="zhipu", debug=False):
    return get_llm_response(TITLE_PROMPT + content, model_type, debug)


def get_llm_response(prompt, model_type="zhipu", debug=False):
    if model_type == "deepseek":
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert in preparing knowledge base materials. You will organize the key points of an article into one coherent paragraph."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            top_p=0.8,
        )
        res = response.choices[0].message.content
    elif model_type == "local":
        import torch  # Only required for local models
        model_name = "internlm/internlm2_5-7b-chat"
        cache_dir = "docs/temp/local_llm"
        os.makedirs(cache_dir, exist_ok=True)
        device = "cuda"

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

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs.input_ids, max_new_tokens=50, max_length=12800, temperature=0.1
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]

        res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": prompt, "temperature": 0.1},
            ],
        )
        res = response.choices[0].message.content

    if debug:
        logger.info(f"Generated result: {res}")

    return res.strip()


def process_file(file_path, model_type, debug=False):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        title = get_title(content, model_type, debug)
        summary = get_summary(content, model_type, debug)
        return {title: summary}


def main():
    parser = argparse.ArgumentParser(description="Extract information from a folder of text files and generate knowledge corpus in JSON format.")
    parser.add_argument("-f", "--folder_path", type=str, required=True, help="Input text folder path")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Output JSON file path")
    parser.add_argument("-m", "--model", type=str, default="zhipu", choices=["zhipu", "deepseek", "local"], help="Model type to use")
    parser.add_argument("-d", "--debug", action="store_true", help="Use debug mode to print the output of each LLM response")

    args = parser.parse_args()

    txt_folder_path = args.folder_path
    output_file_path = args.output_file
    model_type = args.model
    debug = args.debug

    os.makedirs("docs/temp", exist_ok=True)
    error_file_path = "docs/temp/knowledge_error_files.txt"

    filenames = os.listdir(txt_folder_path)
    all_knowledge_data = []

    for filename in tqdm(filenames, desc="Processing files"):
        file_path = os.path.join(txt_folder_path, filename)
        try:
            knowledge_data = process_file(file_path, model_type, debug)
            all_knowledge_data.append(knowledge_data)
        except Exception as e:
            try:
                logger.warning(f"First attempt failed for {filename}, retrying...")
                knowledge_data = process_file(file_path, model_type, debug)
                all_knowledge_data.append(knowledge_data)
            except Exception as e:
                with open(error_file_path, "a", encoding="utf-8") as error_file:
                    logger.error(f"Error processing {file_path}: {e}")
                    error_file.write(file_path + "\n")
                continue

    with open(output_file_path, "w", encoding="utf8") as f:
        json.dump(all_knowledge_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()