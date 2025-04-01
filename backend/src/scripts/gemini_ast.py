import os
import zipfile
import json
import asyncio
import aiohttp
import logging
import google.generativeai as genai
import ast
import javalang
import subprocess
import re
import argparse

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure Google Generative AI with API Key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logging.error("API key not found. Set the GEMINI_API_KEY environment variable.")
    exit(1)

genai.configure(api_key=API_KEY)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
TEMP_FOLDER = os.path.join(BASE_DIR, "temp_extracted")

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".py", ".java", ".js", ".php", ".cpp", ".dart"}

def extract_zip(zip_path):
    logging.info(f"Extracting ZIP: {zip_path}")
    extracted_files = []
    zip_name = os.path.basename(zip_path).replace(".zip", "")
    extract_folder = os.path.join(TEMP_FOLDER, zip_name)
    os.makedirs(extract_folder, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)
            for root, _, files in os.walk(extract_folder):
                for file in files:
                    if any(file.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                        extracted_files.append(os.path.join(root, file))
    except zipfile.BadZipFile:
        logging.error(f"Invalid ZIP file: {zip_path}")
    
    return zip_name, extracted_files

# Language-Specific Parsers

def extract_python_functions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read())
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    except Exception as e:
        logging.error(f"Error parsing Python file {file_path}: {e}")
        return []

def extract_java_functions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        tree = javalang.parse.parse(code)
        
        methods = []
        for class_type in tree.types:
            if isinstance(class_type, javalang.tree.ClassDeclaration):
                methods.extend(method.name for method in class_type.methods)
        
        return methods
    except Exception as e:
        logging.error(f"Error parsing Java file {file_path}: {e}")
        return []

def extract_cpp_functions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        pattern = r"[\w<>]+\s+\w+\s*\([^)]*\)\s*\{[^}]*\}"
        return re.findall(pattern, code, re.DOTALL)
    except Exception as e:
        logging.error(f"Error parsing C++ file {file_path}: {e}")
        return []

def extract_php_functions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        pattern = r"function\s+(\w+)\s*\(.*?\)"
        return re.findall(pattern, code)
    except Exception as e:
        logging.error(f"Error parsing PHP file {file_path}: {e}")
        return []

def extract_dart_functions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        pattern = r"[\w<>]+\s+\w+\s*\([^)]*\)\s*\{"
        return re.findall(pattern, code)
    except Exception as e:
        logging.error(f"Error parsing Dart file {file_path}: {e}")
        return []

async def analyze_code_async(session, functions, filename):
    if not functions:
        logging.warning(f"No functions found in {filename}, skipping analysis.")
        return {"file": filename, "functionality": "[ERROR] No functions found"}

    logging.info(f"Analyzing file: {filename} with {len(functions)} functions")

    prompt = f"""
    Analyze the following class/module `{filename}` and extract its functionalities.
    For each function, provide:
    - functionality_name
    - description
    - input_parameters
    - output_values
    - related_methods
    
    Functions:
    {json.dumps(functions, indent=2)}
    """

    models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-pro-exp"]
    retry_attempts = 3
    delay = 3  

    for model_name in models:
        for attempt in range(retry_attempts):
            try:
                logging.info(f"Using model: {model_name} (Attempt {attempt+1}/{retry_attempts})")
                model = genai.GenerativeModel(model_name)
                response = await asyncio.to_thread(model.generate_content, prompt)

                if response and response.text:
                    return {"file": filename, "functionality": response.text}

                raise Exception(f"Empty response from {model_name}")
            except Exception as e:
                logging.error(f"{model_name} Error for {filename} (Attempt {attempt+1}/{retry_attempts}): {e}")
                if "Resource has been exhausted" in str(e) or "429" in str(e):
                    delay *= 2  
                await asyncio.sleep(delay)

    return {"file": filename, "functionality": "[ERROR] All Gemini models failed after retries"}

async def process_files(zip_name, file_paths):
    output_file = os.path.join(RESULTS_FOLDER, f"{zip_name}.json")

    if os.path.exists(output_file):
        logging.info(f"Skipping {zip_name}, results already exist.")
        return

    async with aiohttp.ClientSession() as session:
        results = []
        semaphore = asyncio.Semaphore(2)  

        async def analyze_file(file_path):
            async with semaphore:
                ext = os.path.splitext(file_path)[1]
                extracted_functions = []

                try:
                    if ext == ".py":
                        extracted_functions = extract_python_functions(file_path)
                    elif ext == ".java":
                        extracted_functions = extract_java_functions(file_path)
                    elif ext == ".cpp":
                        extracted_functions = extract_cpp_functions(file_path)
                    elif ext == ".php":
                        extracted_functions = extract_php_functions(file_path)
                    elif ext == ".dart":
                        extracted_functions = extract_dart_functions(file_path)
                except Exception as e:
                    logging.error(f"Skipping file {file_path} due to extraction error: {e}")
                    return

                if not extracted_functions:
                    logging.warning(f"No functions extracted from {file_path}, skipping analysis.")
                    return

                result = await analyze_code_async(session, extracted_functions, os.path.basename(file_path))
                results.append(result)

        tasks = [analyze_file(file_path) for file_path in file_paths]
        await asyncio.gather(*tasks)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        logging.info(f"✅ Saved results to {output_file}")

async def main(zip_path):
    if not os.path.exists(zip_path):
        logging.error(f"ZIP file does not exist: {zip_path}")
        return
    
    zip_name, extracted_files = extract_zip(zip_path)
    await process_files(zip_name, extracted_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a ZIP file containing source code.")
    parser.add_argument("--file", required=True, help="Path to the uploaded ZIP file.")
    args = parser.parse_args()

    asyncio.run(main(args.file))
