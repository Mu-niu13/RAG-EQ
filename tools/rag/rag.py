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

# New prompt to identify and extract content chunks with context
CONTENT_EXTRACTION_PROMPT = """
Analyze the provided text and identify all meaningful content chunks. These might include:
1. Explicit question-answer pairs
2. Scenarios with context and examples
3. Problems with suggested solutions
4. Situations with advice

For each content chunk, extract:
1. The context (background information, situation description)
2. The problem or question (explicit or implicit)
3. The response, advice, or solution provided

Return the results as a JSON array where each object represents one content chunk, in this format:
[
  {
    "context": "Background information about the situation",
    "problem": "The explicit question or implicit problem described",
    "response": "The advice, solution or response provided"
  }
]

If you encounter examples or scenarios without clear problems/solutions, still extract them by inferring the implicit question.

Here is the text to analyze:
"""

# Enhanced prompt to transform content into a response with both practical advice and emotional intelligence
CONTENT_TRANSFORMATION_PROMPT = """
Transform the following content into a well-structured query and emotionally intelligent response.

Context: {context}
Problem/Situation: {problem}
Advice/Solution: {response}

1. Create a clear, direct question that captures the essence of the problem or situation, incorporating relevant context.
2. Create a comprehensive response that combines practical, actionable suggestions with emotionally intelligent language that acknowledges feelings and concerns.

Format your response as a JSON object with these keys:
- "query": The transformed question ending with a question mark
- "response": The complete emotionally intelligent response that integrates both practical advice and supportive language (about 3-5 sentences)

Ensure your response is a valid, properly formatted JSON object.
"""

# Emotion and topic analysis prompt
ANALYSIS_PROMPT = """
Analyze the following query and response pair and extract:
1. The general topic of the conversation (e.g., "workplace communication", "career development", "relationship advice")
2. The emotional tone of the query (e.g., "anxious", "curious", "frustrated", "uncertain")
3. The emotional tone of the response (e.g., "empathetic", "informative", "encouraging", "supportive")

Query: "{query}"
Response: "{response}"

Format your response as a JSON object with these keys: "topic", "emotion_q", "emotion_r"
Ensure your response is valid JSON with no additional text or explanation.
"""

def get_llm_response(prompt, model_type="zhipu", debug=False, max_retries=2):
    """Get response from LLM models with proper handling for each API and retry logic"""
    for attempt in range(max_retries):
        try:
            if debug and attempt > 0:
                logger.info(f"Retry attempt {attempt+1} for LLM request")
                
            if model_type == "deepseek":
                client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
                )
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are an expert in analyzing text and extracting structured information. Always return valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    top_p=0.8,
                )
                res = response.choices[0].message.content
            elif model_type == "local":
                import torch
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
                    
                modified_prompt = prompt + "\n\nPlease format your response as a valid JSON object only, with no additional text."
                    
                inputs = tokenizer(modified_prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    inputs.input_ids, max_new_tokens=2048, max_length=12800, temperature=0.1
                )
                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(inputs.input_ids, outputs)
                ]

                res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:  # zhipu
                client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
                response = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[
                        {"role": "system", "content": "You are an expert in analyzing text and extracting structured information. Always return valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                res = response.choices[0].message.content

            if debug:
                logger.info(f"Generated raw result: {res[:200]}...")
                
            # If we got a response, attempt to validate it contains JSON before returning
            json_content = extract_json_from_text(res)
            if json_content:
                return json_content
            
            # If we got here, the response didn't contain valid JSON
            if attempt < max_retries - 1:
                # Add more explicit instructions for the retry
                prompt += "\n\nIMPORTANT: Your response MUST be a valid, properly formatted JSON object with no additional explanatory text."
            else:
                # Last attempt failed, return the raw text
                logger.warning(f"Failed to get valid JSON after {max_retries} attempts")
                return res.strip()
                
        except Exception as e:
            logger.error(f"Error in LLM request (attempt {attempt+1}): {str(e)}")
            if attempt == max_retries - 1:
                # Last attempt, raise the exception
                raise
            # Otherwise continue to retry

def extract_json_from_text(text):
    """Extracts and parses JSON from text, handling various formats and patterns."""
    # Try direct parsing first
    try:
        parsed = json.loads(text)
        return text  # Return the original text as it's already valid JSON
    except json.JSONDecodeError:
        pass
    
    # Common patterns to clean up before extraction
    text = text.strip()
    
    # Try to find JSON object
    json_patterns = [
        # Match complete JSON object with possible surrounding text
        (r'({.*})', lambda m: m.group(1)),
        # Match complete JSON array with possible surrounding text
        (r'(\[.*\])', lambda m: m.group(1)),
        # Look for ```json blocks (common in LLM outputs)
        (r'```(?:json)?\s*([\s\S]*?)\s*```', lambda m: m.group(1)),
    ]
    
    import re
    for pattern, extractor in json_patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            potential_json = extractor(matches)
            try:
                # Validate it's parseable
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                continue
    
    return None  # No valid JSON found

def parse_json_safely(text, default_value=None):
    """Parse JSON from text, with improved error handling and extraction."""
    if not text:
        return default_value
        
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON content using regex patterns
    json_content = extract_json_from_text(text)
    if json_content:
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass
    
    # If all extraction methods fail
    logger.error(f"Failed to parse JSON from: {text[:150]}...")
    return default_value

def extract_content_chunks(content, model_type, debug=False):
    """Extract content chunks with context from the text."""
    prompt = CONTENT_EXTRACTION_PROMPT + content
    response = get_llm_response(prompt, model_type, debug)
    
    chunks = parse_json_safely(response, [])
    if not isinstance(chunks, list):
        logger.warning(f"Expected list of chunks but got different format: {type(chunks)}")
        chunks = []
        
    return chunks

def transform_to_query_response(chunk, model_type, debug=False):
    """Transform a content chunk into a query-response pair with an emotionally intelligent combined response.
    
    If the chunk doesn't have an explicit problem, the function will instruct the LLM to infer an implicit
    question based on the provided context and response.
    """
    context = chunk.get("context", "")
    problem = chunk.get("problem", "")
    response = chunk.get("response", "")
    
    if not response:
        logger.warning("Missing response in chunk")
        return None

    # If no explicit problem is provided, adjust the prompt to ask the LLM to generate an implicit question.
    if not problem:
        logger.info("No explicit problem provided; inferring implicit question from context and response")
        prompt = f"""
    Transform the following content into a well-structured query and emotionally intelligent response.

    Context: {context}
    Advice/Solution: {response}

    Since no explicit problem is provided, infer the implicit question suggested by the context and the advice.

    Format your response as a JSON object with these keys:
    - "query": The transformed question ending with a question mark
    - "response": The complete emotionally intelligent response that integrates both practical advice and supportive language (about 3-5 sentences)

    Ensure your response is a valid, properly formatted JSON object.
        """
    else:
        prompt = CONTENT_TRANSFORMATION_PROMPT.format(
            context=context,
            problem=problem,
            response=response
        )
    
    transformation_response = get_llm_response(prompt, model_type, debug)
    transformed = parse_json_safely(transformation_response)
    
    if not transformed or "query" not in transformed or "response" not in transformed:
        logger.warning("Failed to transform chunk to query-response")
        return None
        
    return transformed


def analyze_query_response(query, response, model_type, debug=False):
    """Analyze a query-response pair for topic and emotions."""
    prompt = ANALYSIS_PROMPT.format(query=query, response=response)
    analysis_response = get_llm_response(prompt, model_type, debug)
    
    analysis = parse_json_safely(analysis_response, {})
    if not analysis or not all(k in analysis for k in ["topic", "emotion_q", "emotion_r"]):
        logger.warning("Failed to analyze query-response pair or missing required fields")
        return {"topic": "unknown", "emotion_q": "neutral", "emotion_r": "neutral"}
        
    return analysis

def process_file(file_path, model_type, debug=False):
    """Process a single file and extract all content with analysis."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        
        # Step 1: Extract content chunks with context
        chunks = extract_content_chunks(content, model_type, debug)
        logger.info(f"Extracted {len(chunks)} content chunks from file")
        
        results = []
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Step 2: Transform chunks to query-response format with emotionally intelligent response
                transformed = transform_to_query_response(chunk, model_type, debug)
                if not transformed:
                    logger.warning(f"Skipping chunk {i+1} due to transformation failure")
                    continue
                    
                # Step 3: Analyze for topic and emotions
                analysis = analyze_query_response(
                    transformed["query"], 
                    transformed["response"],
                    model_type, 
                    debug
                )
                
                # Combine everything into the desired format
                conversation_data = {
                    "topic": analysis.get("topic", "unknown"),
                    "query": transformed["query"],
                    "emotion_q": analysis.get("emotion_q", "neutral"),
                    "response": transformed["response"],
                    "emotion_r": analysis.get("emotion_r", "neutral")
                }
                
                results.append(conversation_data)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue
                
        return results

def main():
    parser = argparse.ArgumentParser(description="Extract conversations from text files and generate structured knowledge corpus.")
    parser.add_argument("-f", "--folder_path", type=str, required=True, help="Input text folder path")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Output JSON file path")
    parser.add_argument("-m", "--model", type=str, default="zhipu", choices=["zhipu", "deepseek", "local"], help="Model type to use")
    parser.add_argument("-d", "--debug", action="store_true", help="Use debug mode to print the output of each LLM response")
    parser.add_argument("-b", "--batch_size", type=int, default=5, help="Number of files to process before saving progress")
    parser.add_argument("-c", "--continue_from", action="store_true", help="Continue from where processing was stopped")

    args = parser.parse_args()

    txt_folder_path = args.folder_path
    output_file_path = args.output_file
    model_type = args.model
    debug = args.debug
    batch_size = args.batch_size
    continue_processing = args.continue_from

    # Create necessary directories
    os.makedirs("docs/temp", exist_ok=True)
    error_file_path = "docs/temp/knowledge_error_files.txt"
    progress_file_path = "docs/temp/knowledge_progress.json"

    # Initialize or load progress
    all_knowledge_data = []
    processed_files = set()
    
    if continue_processing and os.path.exists(progress_file_path):
        try:
            with open(progress_file_path, "r", encoding="utf-8") as f:
                all_knowledge_data = json.load(f)
            logger.info(f"Loaded {len(all_knowledge_data)} existing entries from progress file")
            
            # Track processed files without storing in output
            processed_files_path = progress_file_path + ".processed"
            if os.path.exists(processed_files_path):
                with open(processed_files_path, "r", encoding="utf-8") as f:
                    processed_files = set(json.load(f))
                logger.info(f"Found {len(processed_files)} already processed files")
        except Exception as e:
            logger.error(f"Failed to load progress file: {e}")
    
    # Get list of files to process
    filenames = [f for f in os.listdir(txt_folder_path) if os.path.isfile(os.path.join(txt_folder_path, f))]
    files_to_process = [f for f in filenames if f not in processed_files]
    
    logger.info(f"Found {len(filenames)} total files, {len(files_to_process)} remaining to process")
    
    batch_counter = 0
    
    for filename in tqdm(files_to_process, desc="Processing files"):
        file_path = os.path.join(txt_folder_path, filename)
            
        try:
            # Process file to extract content
            conversation_data_list = process_file(file_path, model_type, debug)
            
            if conversation_data_list:
                # Add processed items to the results
                all_knowledge_data.extend(conversation_data_list)
                
                # Track processed files internally
                processed_files.add(filename)
                logger.info(f"Successfully extracted {len(conversation_data_list)} conversations from {filename}")
            else:
                logger.warning(f"No valid conversations extracted from {filename}")
                # Still mark as processed to avoid repeated failures
                processed_files.add(filename)
            
            # Save progress periodically
            batch_counter += 1
            if batch_counter >= batch_size:
                with open(progress_file_path, "w", encoding="utf-8") as f:
                    json.dump(all_knowledge_data, f, ensure_ascii=False, indent=2)
                # Save processed files list separately
                with open(progress_file_path + ".processed", "w", encoding="utf-8") as f:
                    json.dump(list(processed_files), f, ensure_ascii=False)
                logger.info(f"Saved progress after processing {batch_counter} files, total entries: {len(all_knowledge_data)}")
                batch_counter = 0
                
        except Exception as e:
            try:
                logger.warning(f"First attempt failed for {filename}, retrying once more: {str(e)}")
                conversation_data_list = process_file(file_path, model_type, debug)
                
                if conversation_data_list:
                    # Add processed items to results
                    all_knowledge_data.extend(conversation_data_list)
                    
                    # Track processed files internally
                    processed_files.add(filename)
            except Exception as e:
                with open(error_file_path, "a", encoding="utf-8") as error_file:
                    error_message = f"{file_path} - {str(e)}"
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    error_file.write(f"{error_message}\n")
                # Mark as processed to avoid getting stuck in a loop
                processed_files.add(filename)
                continue

    # Save final output
    if all_knowledge_data:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(all_knowledge_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved {len(all_knowledge_data)} total entries to {output_file_path}")
    else:
        logger.warning("No data was extracted. Output file not created.")
    
    # Clean up progress files if successful
    if os.path.exists(progress_file_path) and len(processed_files) == len(filenames):
        os.remove(progress_file_path)
        if os.path.exists(progress_file_path + ".processed"):
            os.remove(progress_file_path + ".processed")
        logger.info("Removed progress files after successful completion")
        
    logger.info(f"Processing complete: {len(processed_files)} files processed, {len(all_knowledge_data)} total entries extracted")
    if os.path.exists(error_file_path):
        with open(error_file_path, "r") as f:
            error_count = len(f.readlines())
        logger.warning(f"{error_count} files had errors during processing. See {error_file_path} for details.")

if __name__ == "__main__":
    main()

