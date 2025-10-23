import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from collections import defaultdict
from io import StringIO
from typing import Dict, List
import concurrent.futures
import logging
import re
import ast
import gc
import multiprocessing
from multiprocessing import Manager

logger = logging.getLogger("function call")
SINGLE_CASE_EXEC_TIMEOUT = 6

def extract_python_code(text, min_length=20, strict_syntax=True):
    code_pattern = r"(?i)```(?:python|py)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = []
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) < min_length:
            continue

        # verify code syntax
        if strict_syntax:
            try:
                ast.parse(clean_block, mode="exec")
            except (SyntaxError, IndentationError):
                continue

        valid_blocks.append(clean_block)

    if not valid_blocks:
        return None
    # return the last code block
    return valid_blocks[-1]

def call_verify(problem, generation, debug, timeout=SINGLE_CASE_EXEC_TIMEOUT):

    tmp_id = str(uuid.uuid4())
    input_data = {
        "sample": problem,
        "test": extract_python_code(generation),
        "debug": debug,
        "timeout": timeout,
    }
    with open(f"/tmp/{tmp_id}-input.json", "w") as temp_file:
        json.dump(input_data, temp_file)
    start_time = time.time()

    venv_python = "python3"
    if debug:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    else:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
    pro = subprocess.Popen(
        " ".join(
            [
                venv_python,
                "marti/verifiers/areal/function/testing_util.py",
                "--tmp_id",
                tmp_id,
            ]
        ),
        shell=True,
        preexec_fn=os.setsid,
        stdout=stdout,
        stderr=stderr,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
    )
    try:
        if not debug:
            pro.wait(600)
        else:
            out, err = pro.communicate(timeout=600)
    except Exception as e:
        pass
    if debug:
        logger.info(f"Process return code: {pro.returncode}")
        logger.info(f"STDOUT: {out.decode()}")
        logger.info(f"STDERR: {err.decode()}")
    try:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    result = {"result": [False], "info": {}}
    try:
        with open(f"/tmp/{tmp_id}-output.json", "r") as f:
            result = json.load(f)
    except FileNotFoundError as e:
        logger.warning(
            f"Failed to parse generated answers. FileNotFoundError. Set 0 reward."
        )
    except Exception as e:
        logger.warning(
            f"Failed to parse generated answers. {e}. Set 0 reward."
        )
    finally:
        if os.path.exists(f"/tmp/{tmp_id}-input.json"):
            os.remove(f"/tmp/{tmp_id}-input.json")
        if os.path.exists(f"/tmp/{tmp_id}-output.json"):
            os.remove(f"/tmp/{tmp_id}-output.json")
        del input_data, pro
        gc.collect()
    execution_time = time.time() - start_time
    # logger.info(
    #     f'[call_verify] start_time: {str(start_time)}, Time elapsed: {execution_time * 1000:.0f} ms'
    # )
    return result["result"], result["info"]

def code_verify(problems, generateds, debug=False, return_metadata=False):
    # assert len(generateds) == len(query_ids)
    # problems = [id2info[qid] for qid in query_ids]

    # final_results = []

    # infer_args = []
    # for query_id, generated, problem in zip(query_ids, generateds, problems):
    #     infer_args.append((problem, generated, debug, SINGLE_CASE_EXEC_TIMEOUT))

    # run_results = []
    # num_process = max(1, os.cpu_count() // 8)
    # with concurrent.futures.ProcessPoolExecutor(num_process) as executor:
    #     run_results = executor.map(call_verify, *zip(*infer_args))
    # print(f"[code_verify] problems: {problems}")
    # print(f"[code_verify] generateds: {generateds}")
    # infer_args = []
    # infer_args.append((problems, generateds, debug, SINGLE_CASE_EXEC_TIMEOUT))

    def evaluate_code(tests, generation, debug, test_results, test_fn):
        """Helper function to run tests in separate process."""
        try:
            test_results.append(test_fn(tests, generation, debug, SINGLE_CASE_EXEC_TIMEOUT))
        except Exception as e:
            print(f"Error in evaluate_code: {e}")
    manager = Manager()
    test_results = manager.list()
    process = multiprocessing.Process(
        target=evaluate_code,
        args=(problems, generateds, debug, test_results, call_verify)
    )
    process.start()
    if isinstance(problems, list):
        num_test = len(problems)
    elif isinstance(problems, str):
        num_test = len(json.loads(problems)["inputs"])
    elif isinstance(problems, dict):
        num_test = len(problems['inputs'] if isinstance(problems['inputs'], list) else json.loads(problems["inputs"]))
    hard_limit = SINGLE_CASE_EXEC_TIMEOUT * num_test + 5
    process.join(timeout=hard_limit)

    if process.is_alive():
        process.kill()
    run_results = test_results[:]
    if len(test_results) == 0:
        run_result = ([False], {})
    #assert len(test_results) == 1, f"Expected exactly one test result, but got {test_results}"
    else:
        run_result = run_results[0]
    # curr_res, metadata = run_result
    # run_results = [r==True for r in run_results]


    # run_results = []
    # num_process = os.cpu_count()
    # with concurrent.futures.ProcessPoolExecutor(num_process) as executor:
    #     run_results = executor.map(call_verify, *zip(*infer_args))
    # run_result = run_results[0]
    # for run_result in run_results:
    curr_res, metadata = run_result
    print(f"[code_verify] metadata: {metadata}")
    print(f"[code_verify] curr_res: {curr_res}")

    if return_metadata:
        return run_result

    if any(x != True for x in curr_res):
        return 0
    else:
        return 1