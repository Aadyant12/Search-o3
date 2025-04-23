# run_search_o1.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict
import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer, util

# from bing_search import (
#     bing_web_search, 
#     extract_relevant_info, 
#     fetch_page_content, 
#     extract_snippet_with_context
# )
from google_search import (
    google_web_search,
    extract_relevant_info,
    fetch_page_content,
    extract_snippet_with_context,
)
from evaluate import (
    run_evaluation, 
    extract_answer
)
from prompts import (
    get_gpqa_search_o1_instruction, 
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction, 
    get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search O1 for various datasets and models.")

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle'],
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['test', 'diamond', 'main', 'extended'],
        help="Dataset split to use."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=10,
        help="Maximum number of searches per question."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=15,
        help="Maximum number of turns."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Maximum number of search documents to return."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    parser.add_argument(
        '--use_jina',
        type=bool,
        default=True,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    # # Bing API Configuration
    # parser.add_argument(
    #     '--bing_subscription_key',
    #     type=str,
    #     required=True,
    #     help="Bing Search API subscription key."
    # )

    # parser.add_argument(
    #     '--bing_endpoint',
    #     type=str,
    #     default="https://api.bing.microsoft.com/v7.0/search",
    #     help="Bing Search API endpoint."
    # )

    # Google API Configuration
    parser.add_argument(
        '--google_subscription_key',
        type=str,
        required=True,
        help="Google Search API key."
    )

    # Enhanced HyDE configuration
    parser.add_argument(
        '--use_hyde',
        action='store_true',
        help="Whether to use Hypothetical Document Embeddings for search enhancement."
    )
    
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help="Embedding model to use for HyDE."
    )
    
    parser.add_argument(
        '--hyde_temperature',
        type=float,
        default=0.7,
        help="Temperature for generating hypothetical documents."
    )
    
    parser.add_argument(
        '--num_hyde_docs',
        type=int,
        default=3,
        help="Number of hypothetical documents to generate for ensemble."
    )
    
    parser.add_argument(
        '--context_aware_hyde',
        action='store_true',
        help="Whether to use context-aware HyDE."
    )
    
    parser.add_argument(
        '--adaptive_ensemble',
        action='store_true',
        help="Whether to use adaptive ensemble of hypothetical documents."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Extract arguments
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    MAX_SEARCH_LIMIT = args.max_search_limit
    MAX_TURN = args.max_turn
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    # bing_subscription_key = args.bing_subscription_key
    google_subscription_key = args.google_subscription_key
    # bing_endpoint = args.bing_endpoint
    use_jina = args.use_jina
    jina_api_key = args.jina_api_key
    
    # Extract HyDE arguments
    use_hyde = args.use_hyde
    embedding_model_name = args.embedding_model
    hyde_temperature = args.hyde_temperature
    num_hyde_docs = args.num_hyde_docs
    context_aware_hyde = args.context_aware_hyde
    adaptive_ensemble = args.adaptive_ensemble
    
    # Adjust parameters based on dataset
    if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        MAX_SEARCH_LIMIT = 5
        if dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:
            MAX_SEARCH_LIMIT = 10
            MAX_TURN = 15
        top_k = 10
        max_doc_len = 3000
    
    if args.jina_api_key == 'None':
        jina_api_key = None

    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0

    # Data paths based on dataset
    if dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = '/content/drive/MyDrive/search-o3/Search-o3/data/GPQA/diamond.json'
    else:
        data_path = f'./data/QA_Datasets/{dataset_name}.json'

    print('-----------------------')
    print(f'Using {dataset_name} {split} set.')
    print('-----------------------')

    # ---------------------- Caching Mechanism ----------------------
    # Define cache directories and file paths
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, 'search_cache.json')
    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache = json.load(f)
    else:
        search_cache = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)
    else:
        url_cache = {}

    # Function to save caches
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # ---------------------- Model Loading ----------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Define output directory based on model and dataset
    if 'qwq' in model_path.lower():
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.qwq.search_o1'
            if dataset_name == 'gpqa' and (MAX_SEARCH_LIMIT != 5 or top_k != 10):
                output_dir = f'./outputs/runs.analysis/{dataset_name}.qwq.search_o1.{MAX_SEARCH_LIMIT}.{top_k}'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.search_o1'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.search_o1'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        dtype="float16",
        enforce_eager=True,
        max_cpu_loras=1,  # Enable CPU offloading 
        quantization="gptq", 
    )

    # Initialize embedding model if HyDE is enabled
    if use_hyde:
        print(f"Initializing embedding model {embedding_model_name} for Weighted HyDE...")
        embedding_model = SentenceTransformer(embedding_model_name)
        print(f"Context-aware HyDE: {context_aware_hyde}")
        print(f"Adaptive ensemble: {adaptive_ensemble}")
        print(f"Number of hypothetical documents: {num_hyde_docs}")

    # ---------------------- Data Loading ----------------------
    with open(data_path, 'r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)

    # ---------------------- Batch Generation Function ----------------------
    def generate_webpage_to_reasonchain_batch(
        original_questions: List[str],
        prev_reasonings: List[str],
        search_queries: List[str],
        documents: List[str],
        dataset_name: str,
        batch_output_records: List[Dict],  # New parameter to collect outputs
        max_tokens: int = 32768,
        coherent: bool = False,
    ) -> List[str]:
        user_prompts = [
            get_webpage_to_reasonchain_instruction(r, sq, doc)
            for r, sq, doc in zip(prev_reasonings, search_queries, documents)
        ]

        prompts = [{"role": "user", "content": up} for up in user_prompts]
        prompts = [tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]

        output = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
            )
        )

        raw_outputs = [out.outputs[0].text for out in output]
        extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]

        for i, (p, r, e) in enumerate(zip(prompts, raw_outputs, extracted_infos)):
            batch_output_records.append({
                'prompt': p,
                'raw_output': r,
                'extracted_info': e
            })

        return extracted_infos

    # ---------------------- Preparation of Input Prompts ----------------------
    input_list = []
    for item in filtered_data:
        question = item['Question']

        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            if dataset_name in ['nq', 'triviaqa']:
                instruction = get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            elif dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:
                instruction = get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            instruction = get_math_search_o1_instruction(MAX_SEARCH_LIMIT)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name == 'gpqa':
            instruction = get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)

        elif dataset_name == 'livecode':
            instruction = get_code_search_o1_instruction(MAX_SEARCH_LIMIT)
            question_title = item.get('question_title', '')
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            user_prompt = ""  # Default to empty if dataset not matched

        prompt = [{"role": "user", "content": instruction + user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)

    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]

    # Initialize active sequences
    active_sequences = [{
        'item': item,
        'prompt': prompt,
        'output': '',
        'finished': False,
        'history': [],
        'search_count': 0,
        'executed_search_queries': set(),
    } for item, prompt in zip(filtered_data, input_list)]

    # ---------------------- Set Max Tokens ----------------------
    if 'qwq' in model_path.lower():
        if dataset_name in ['aime', 'amc', 'livecode']:
            max_tokens = 32768
        else:
            max_tokens = 20480
    else:
        max_tokens = 8192

    # ---------------------- Generation Function ----------------------
    def run_generation(sequences: List[Dict], max_tokens: int) -> List:
        batch_size = 32  # Choose a smaller batch size
        all_outputs = []
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_prompts = [s['prompt'] for s in batch_sequences]
            
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k_sampling,
                repetition_penalty=repetition_penalty,
                stop=[END_SEARCH_QUERY, tokenizer.eos_token],
                include_stop_str_in_output=True,
            )
            
            print(f"Processing batch {i//batch_size + 1}/{(len(sequences) + batch_size - 1)//batch_size} ({len(batch_prompts)} sequences)")
            batch_outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
            all_outputs.extend(batch_outputs)
            
            # Optional: Clear cache between batches
            torch.cuda.empty_cache()
            
        return all_outputs  

    # Function to extract text between two tags
    def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def replace_recent_steps(origin_str, replace_str):
        """
        Replaces specific steps in the original reasoning steps with new steps.
        If a replacement step contains "DELETE THIS STEP", that step is removed.

        Parameters:
        - origin_str (str): The original reasoning steps.
        - replace_str (str): The steps to replace or delete.

        Returns:
        - str: The updated reasoning steps after applying replacements.
        """

        def parse_steps(text):
            """
            Parses the reasoning steps from a given text.

            Parameters:
            - text (str): The text containing reasoning steps.

            Returns:
            - dict: A dictionary mapping step numbers to their content.
            """
            step_pattern = re.compile(r"Step\s+(\d+):\s*")
            steps = {}
            current_step_num = None
            current_content = []

            for line in text.splitlines():
                step_match = step_pattern.match(line)
                if step_match:
                    # If there's an ongoing step, save its content
                    if current_step_num is not None:
                        steps[current_step_num] = "\n".join(current_content).strip()
                    current_step_num = int(step_match.group(1))
                    content = line[step_match.end():].strip()
                    current_content = [content] if content else []
                else:
                    if current_step_num is not None:
                        current_content.append(line)
            
            # Save the last step if any
            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            
            return steps

        # Parse the original and replacement steps
        origin_steps = parse_steps(origin_str)
        replace_steps = parse_steps(replace_str)

        # Apply replacements
        for step_num, content in replace_steps.items():
            if "DELETE THIS STEP" in content:
                # Remove the step if it exists
                if step_num in origin_steps:
                    del origin_steps[step_num]
            else:
                # Replace or add the step
                origin_steps[step_num] = content

        # Sort the steps by step number
        sorted_steps = sorted(origin_steps.items())

        # Reconstruct the reasoning steps as a single string
        new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

        return new_reasoning_steps

    # ---------------------- Initialize Collection Structure ----------------------
    # Initialize a list to collect batch outputs
    batch_output_records = []

    start_time = time.time()
    turn = 0

    # Main loop until all sequences are finished or maximum turns reached
    while True:
        # Identify sequences that need generation
        sequences_needing_generation = [seq for seq in active_sequences if not seq['finished']]

        if sequences_needing_generation:
            turn += 1
            print(f'\n-------------- Turn {turn} --------------')
            print(f"We have {len(sequences_needing_generation)} sequences needing generation...")
            outputs = run_generation(sequences_needing_generation, max_tokens)
            print("Generation completed, processing outputs...")

            # Initialize batch variables
            batch_relevant_info = []
            batch_original_questions = []
            batch_prev_reasonings = []
            batch_search_queries = []
            batch_documents = []
            batch_sequences = []

            # Collect URLs to fetch across all sequences
            all_urls_to_fetch = set()
            url_snippets = {}
            url_sequence_map = {}  # Map URL to list of sequences needing it

            # Process each sequence and collect URLs
            for seq, out in zip(sequences_needing_generation, outputs):
                text = out.outputs[0].text
                seq['history'].append(text)
                # Append generated text to prompt and output
                seq['prompt'] += text
                seq['output'] += text

                # Extract search query
                search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                # If a search query is present and needs to be executed
                if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                    if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']:
                        # Execute search, use cache if available
                        cache_key = f"{search_query}_{use_hyde}_{context_aware_hyde}_{adaptive_ensemble}"
                        if cache_key in search_cache:
                            results = search_cache[cache_key]
                            print(f"Using cached search results for query: \"{search_query}\"")
                        else:
                            try:
                                if use_hyde:
                                    # Extract reasoning context for context-aware HyDE
                                    reasoning_context = seq['output']
                                    
                                    results = enhanced_google_search(
                                        search_query,
                                        reasoning_context,
                                        google_subscription_key, 
                                        llm=llm,
                                        tokenizer=tokenizer,
                                        embedding_model=embedding_model,
                                        use_hyde=use_hyde,
                                        context_aware=context_aware_hyde,
                                        adaptive_ensemble=adaptive_ensemble,
                                        num_docs=num_hyde_docs,
                                        hyde_temperature=hyde_temperature,
                                        top_k=top_k
                                    )
                                else:
                                    results = google_web_search(search_query, google_subscription_key)
                                
                                search_cache[cache_key] = results
                                print(f"Executed and cached search for query: \"{search_query}\"")
                            except Exception as e:
                                print(f"Error during search query '{search_query}': {e}")
                                search_cache[cache_key] = {}
                                results = {}

                        # Extract relevant information from Google search results
                        relevant_info = extract_relevant_info(results)[:top_k]
                        seq['relevant_info'] = relevant_info

                        # Extract URLs and snippets
                        urls_to_fetch = [it['link'] for it in relevant_info]
                        snippets = {info['link']: info['snippet'] for info in relevant_info if 'snippet' in info}

                        # Filter URLs that are not cached
                        urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
                        cached_urls = [u for u in urls_to_fetch if u in url_cache]

                        # Store info for all_urls_to_fetch and url_snippets
                        for url in urls_to_fetch_filtered:
                            all_urls_to_fetch.add(url)
                            url_snippets[url] = snippets.get(url, "")

                        all_reasoning_steps = seq['output']
                        all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")

                        truncated_prev_reasoning = ""
                        for i, step in enumerate(all_reasoning_steps):
                            truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                        prev_steps = truncated_prev_reasoning.split('\n\n')
                        if len(prev_steps) <= 5:
                            truncated_prev_reasoning = '\n\n'.join(prev_steps)
                        else:
                            truncated_prev_reasoning = ''
                            for i, step in enumerate(prev_steps):
                                if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                                    truncated_prev_reasoning += step + '\n\n'
                                else:
                                    if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                        truncated_prev_reasoning += '...\n\n'
                        truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

                        # Collect parameters for batch processing
                        batch_relevant_info.append(relevant_info)
                        batch_original_questions.append(seq['item']['Question'])
                        batch_prev_reasonings.append(truncated_prev_reasoning)
                        batch_search_queries.append(search_query)
                        batch_sequences.append(seq)

                        # Update search count and executed queries
                        seq['search_count'] += 1
                        seq['executed_search_queries'].add(search_query)

                    elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Search limit reached for query: \"{search_query}\"")

                    elif search_query in seq['executed_search_queries']:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Repeated search for query: \"{search_query}\"")

                else:
                    # If no search query needs to be executed, mark the sequence as finished
                    seq['finished'] = True
                    print("Sequence marked as complete.")

            # Batch fetch all URLs at once to optimize speed
            if all_urls_to_fetch:
                print(f"Fetching {len(all_urls_to_fetch)} URLs...")
                try:
                    fetched_contents = fetch_page_content(
                        list(all_urls_to_fetch),
                        use_jina=use_jina,
                        jina_api_key=jina_api_key,
                        # snippets=url_snippets  # Do not pass snippets when updating url_cache directly
                    )
                    print(f"Fetched {len(fetched_contents)} URLs successfully.")
                except Exception as e:
                    print(f"Error during batch URL fetching: {e}")
                    fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
                # Update cache with fetched contents
                for url, content in fetched_contents.items():
                    url_cache[url] = content

            # After fetching, prepare formatted documents for batch processing
            for relevant_info in batch_relevant_info:
                formatted_documents = ""
                for i, doc_info in enumerate(relevant_info):
                    url = doc_info['link']
                    raw_context = url_cache.get(url, "")
                    doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')            
                    success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=max_doc_len)
                    if success:
                        context = filtered_context
                    else:
                        context = raw_context[:max_doc_len*2]

                    doc_info['context'] = context
                    formatted_documents += f"**Web Page {i + 1}:**\n"
                    formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
                    
                batch_documents.append(formatted_documents)

            # After fetching, prepare for batch processing if there are any
            if batch_sequences:
                print(f"Batch processing {len(batch_sequences)} sequences with generate_webpage_to_reasonchain_batch...")
                webpage_analyses = generate_webpage_to_reasonchain_batch(
                    original_questions=batch_original_questions,
                    prev_reasonings=batch_prev_reasonings,
                    search_queries=batch_search_queries,
                    documents=batch_documents,
                    dataset_name=dataset_name,
                    batch_output_records=batch_output_records,  # Pass the collection list
                    max_tokens=max_tokens,
                )
                print("Batch generation completed, assigning outputs to sequences...")

                for seq, analysis in zip(batch_sequences, webpage_analyses):
                    if isinstance(analysis, str):
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                    else:
                        append_text = replace_recent_steps(seq['output'], analysis)
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)

        # Check if all sequences are finished
        unfinished = [seq for seq in active_sequences if not seq['finished']]
        if not unfinished:
            break
        else:
            if turn >= MAX_TURN:
                print(f"Maximum number of turns ({MAX_TURN}) reached, stopping.")
                break

    total_time = time.time() - start_time

    # ---------------------- Save Batch Output Records to JSON File ----------------------
    # Define output JSON file path
    t = time.localtime()
    batch_output_file = os.path.join(output_dir, f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json')

    # Save batch_output_records to JSON file
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_output_records, f, ensure_ascii=False, indent=2)

    print(f"Batch outputs saved to {batch_output_file}")

    # Prepare output list for evaluation
    output_list = [seq['output'] for seq in active_sequences]

    # Run evaluation
    run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split)

    # ---------------------- Update Search and URL Cache ----------------------
    print('Updating Search and URL Cache...')
    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache_new = json.load(f)
    else:
        search_cache_new = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache_new = json.load(f)
    else:
        url_cache_new = {}

    search_cache.update(search_cache_new)
    url_cache.update(url_cache_new)

    save_caches()

    print("Process completed.")

def extract_uncertainty(reasoning_context):
    """
    Extract uncertainty markers and surrounding context from the reasoning chain.
    
    Args:
        reasoning_context: The current reasoning chain
        
    Returns:
        A list of uncertainty contexts
    """
    # Define uncertainty markers
    uncertainty_markers = [
        "perhaps", "likely", "possibly", "might", "may", "could", 
        "uncertain", "not sure", "unclear", "don't know", "unknown",
        "need more information", "need to verify", "need to check"
    ]
    
    # Find sentences containing uncertainty markers
    sentences = re.split(r'(?<=[.!?])\s+', reasoning_context)
    uncertainty_contexts = []
    
    for sentence in sentences:
        if any(marker in sentence.lower() for marker in uncertainty_markers):
            # Get surrounding context (previous and next sentence if available)
            idx = sentences.index(sentence)
            start_idx = max(0, idx - 1)
            end_idx = min(len(sentences), idx + 2)
            context = " ".join(sentences[start_idx:end_idx])
            uncertainty_contexts.append(context)
    
    # If no uncertainty markers found, return the last few sentences as context
    if not uncertainty_contexts and len(sentences) > 0:
        uncertainty_contexts = [" ".join(sentences[-3:])]
    
    return uncertainty_contexts

def generate_context_aware_hypothetical_document(llm, query, reasoning_context, tokenizer, temperature=0.7):
    """
    Generate a hypothetical document based on the query and reasoning context.
    
    Args:
        llm: The language model
        query: The search query
        reasoning_context: The current reasoning chain
        tokenizer: Tokenizer for the language model
        temperature: Temperature for generation
        
    Returns:
        A hypothetical document that addresses the knowledge gap
    """
    # Extract uncertainty contexts
    uncertainty_contexts = extract_uncertainty(reasoning_context)
    uncertainty_text = "\n".join(uncertainty_contexts)
    
    # Create a prompt that incorporates both the query and reasoning context
    hyde_prompt = f"""You are an expert researcher. Generate a detailed document that would be the perfect search result for addressing the knowledge gaps in the current reasoning.

Current reasoning context with uncertainty:
{uncertainty_text}

Search query: {query}

Generate a detailed document that would provide the missing information:"""
    
    prompt = [{"role": "user", "content": hyde_prompt}]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
    )
    
    output = llm.generate(prompt, sampling_params=sampling_params)
    hypothetical_doc = output.outputs[0].text.strip()
    
    return hypothetical_doc

def evaluate_document_confidence(doc, query, reasoning_context, llm, tokenizer):
    """
    Evaluate the confidence score for a hypothetical document.
    
    Args:
        doc: The hypothetical document
        query: The search query
        reasoning_context: The current reasoning chain
        llm: The language model
        tokenizer: Tokenizer for the language model
        
    Returns:
        A confidence score between 0 and 1
    """
    eval_prompt = f"""On a scale of 0 to 10, rate how well the following document addresses the knowledge gaps in the reasoning context and answers the search query.

Search query: {query}

Current reasoning context (abbreviated):
{reasoning_context[:500]}...

Document to evaluate:
{doc[:1000]}...

Provide your rating as a single number between 0 and 10, where 0 means completely irrelevant and 10 means perfectly addresses the knowledge gap."""
    
    prompt = [{"role": "user", "content": eval_prompt}]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.1,
        top_p=0.95,
    )
    
    output = llm.generate(prompt, sampling_params=sampling_params)
    response = output.outputs[0].text.strip()
    
    # Extract the numeric rating
    try:
        # Try to find a number in the response
        match = re.search(r'(\d+(\.\d+)?)', response)
        if match:
            rating = float(match.group(1))
            # Normalize to 0-1 range
            confidence = min(max(rating / 10.0, 0.0), 1.0)
        else:
            confidence = 0.5  # Default if no number found
    except:
        confidence = 0.5  # Default if parsing fails
    
    return confidence

def generate_weighted_hyde_embeddings(query, reasoning_context, llm, tokenizer, embedding_model, 
                                     num_docs=3, temperature=0.7, context_aware=True, adaptive_ensemble=True):
    """
    Generate weighted HyDE embeddings using context-aware documents and adaptive ensemble.
    
    Args:
        query: The search query
        reasoning_context: The current reasoning chain
        llm: The language model
        tokenizer: Tokenizer for the language model
        embedding_model: Model for embedding documents
        num_docs: Number of hypothetical documents to generate
        temperature: Temperature for generation
        context_aware: Whether to use context-aware HyDE
        adaptive_ensemble: Whether to use adaptive ensemble
        
    Returns:
        A weighted ensemble embedding
    """
    hypothetical_docs = []
    
    # Generate multiple hypothetical documents
    for i in range(num_docs):
        if context_aware:
            doc = generate_context_aware_hypothetical_document(
                llm, query, reasoning_context, tokenizer, temperature=temperature
            )
        else:
            # Fall back to standard HyDE if context_aware is False
            hyde_prompt = f"Generate a detailed document that would be the perfect search result for this query: {query}"
            prompt = [{"role": "user", "content": hyde_prompt}]
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            
            sampling_params = SamplingParams(
                max_tokens=1024,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
            )
            
            output = llm.generate(prompt, sampling_params=sampling_params)
            doc = output.outputs[0].text.strip()
        
        hypothetical_docs.append(doc)
    
    # Compute embeddings for all documents
    embeddings = embedding_model.encode(hypothetical_docs)
    
    if adaptive_ensemble and len(hypothetical_docs) > 1:
        # Compute confidence scores for adaptive ensemble
        confidence_scores = []
        for doc in hypothetical_docs:
            confidence = evaluate_document_confidence(doc, query, reasoning_context, llm, tokenizer)
            confidence_scores.append(confidence)
        
        # Normalize confidence scores to sum to 1
        total_confidence = sum(confidence_scores)
        if total_confidence > 0:
            weights = [score / total_confidence for score in confidence_scores]
        else:
            weights = [1.0 / len(confidence_scores)] * len(confidence_scores)
        
        # Compute weighted average embedding
        weighted_embedding = np.zeros_like(embeddings[0])
        for i, embedding in enumerate(embeddings):
            weighted_embedding += weights[i] * embedding
        
        # Normalize the weighted embedding
        weighted_embedding = weighted_embedding / np.linalg.norm(weighted_embedding)
        
        return weighted_embedding, hypothetical_docs, weights
    else:
        # If not using adaptive ensemble, just average the embeddings
        average_embedding = np.mean(embeddings, axis=0)
        average_embedding = average_embedding / np.linalg.norm(average_embedding)
        weights = [1.0 / len(hypothetical_docs)] * len(hypothetical_docs)
        
        return average_embedding, hypothetical_docs, weights

def enhanced_google_search(query, reasoning_context, google_subscription_key, 
                          llm=None, tokenizer=None, embedding_model=None,
                          use_hyde=False, context_aware=True, adaptive_ensemble=True,
                          num_docs=3, hyde_temperature=0.7, top_k=10):
    """
    Perform enhanced Google search with Weighted HyDE.
    
    Args:
        query: The search query
        reasoning_context: The current reasoning chain
        google_subscription_key: Google API key
        llm: Language model for generating hypothetical documents
        tokenizer: Tokenizer for the language model
        embedding_model: Model for embedding documents
        use_hyde: Whether to use HyDE
        context_aware: Whether to use context-aware HyDE
        adaptive_ensemble: Whether to use adaptive ensemble
        num_docs: Number of hypothetical documents to generate
        hyde_temperature: Temperature for generating hypothetical documents
        top_k: Maximum number of search results to return
        
    Returns:
        Search results
    """
    if not use_hyde:
        return google_web_search(query, google_subscription_key)
    
    # Generate weighted HyDE embeddings
    weighted_embedding, hypothetical_docs, weights = generate_weighted_hyde_embeddings(
        query, reasoning_context, llm, tokenizer, embedding_model,
        num_docs=num_docs, temperature=hyde_temperature,
        context_aware=context_aware, adaptive_ensemble=adaptive_ensemble
    )
    
    # Log the hypothetical documents and their weights for debugging
    print(f"Generated {len(hypothetical_docs)} hypothetical documents with weights: {weights}")
    
    # Extract key information from hypothetical documents to create enhanced queries
    enhanced_queries = [query]  # Start with the original query
    
    # Extract important sentences from hypothetical documents, weighted by confidence
    for doc, weight in zip(hypothetical_docs, weights):
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', doc)
        
        # Select sentences based on weight (more sentences from higher-weighted docs)
        num_sentences = max(1, int(round(3 * weight)))  # At least 1, up to 3 sentences based on weight
        
        if len(sentences) <= num_sentences:
            selected_sentences = sentences
        else:
            # Simple heuristic: longer sentences might contain more information
            selected_sentences = sorted(sentences, key=len, reverse=True)[:num_sentences]
        
        enhanced_queries.extend(selected_sentences)
    
    # Limit the number of queries to avoid excessive API calls
    enhanced_queries = enhanced_queries[:5]  # Limit to 5 queries total
    
    # Perform search for each query
    all_results = []
    for q in enhanced_queries:
        results = google_web_search(q, google_subscription_key)
        if 'items' in results:  # Google Search API returns 'items' instead of 'webPages'
            all_results.extend(results['items'])
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_results = []
    for result in all_results:
        if 'link' in result and result['link'] not in seen_urls:  # Google uses 'link' instead of 'url'
            seen_urls.add(result['link'])
            unique_results.append(result)
    
    # Format results to match the expected structure for Google Search API
    combined_results = {
        'items': unique_results[:top_k]  # Limit to top_k results
    }
    
    return combined_results

if __name__ == "__main__":
    main()
