"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI and Anthropic Claude APIs.
"""
import json
import random
import openai
import time

from utils import *

# Initialize API clients
openai.api_key = openai_api_key

# Import Anthropic if available
try:
    import anthropic
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if 'anthropic_api_key' in dir() else None
except ImportError:
    anthropic_client = None

# Get LLM provider from utils (default to openai if not set)
LLM_PROVIDER = llm_provider if 'llm_provider' in dir() else "openai"

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt):
  temp_sleep()

  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
  )
  return completion["choices"][0]["message"]["content"]


# ============================================================================
# #####################[SECTION 1: CLAUDE API STRUCTURE] #####################
# ============================================================================

def Claude_request(prompt):
  """
  Given a prompt, make a request to Anthropic Claude API and return the response.
  Uses Claude 3.5 Sonnet by default.
  ARGS:
    prompt: a str prompt
  RETURNS:
    a str of Claude's response.
  """
  temp_sleep()

  try:
    if anthropic_client is None:
      return "CLAUDE ERROR: Anthropic client not initialized"

    message = anthropic_client.messages.create(
      model="claude-3-5-sonnet-20241022",
      max_tokens=2000,
      messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

  except Exception as e:
    print(f"CLAUDE ERROR: {str(e)}")
    return "CLAUDE ERROR"


def Claude_Opus_request(prompt):
  """
  Given a prompt, make a request to Anthropic Claude API using Opus model.
  ARGS:
    prompt: a str prompt
  RETURNS:
    a str of Claude's response.
  """
  temp_sleep()

  try:
    if anthropic_client is None:
      return "CLAUDE ERROR: Anthropic client not initialized"

    message = anthropic_client.messages.create(
      model="claude-3-opus-20240229",
      max_tokens=2000,
      messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

  except Exception as e:
    print(f"CLAUDE ERROR: {str(e)}")
    return "CLAUDE ERROR"


# ============================================================================
# ##################[SECTION 2: CHATGPT-3/GPT-4 STRUCTURE] ###################
# ============================================================================

def GPT4_request(prompt):
  """
  Given a prompt, make a request to the configured LLM provider.
  Routes to either GPT-4 or Claude Opus based on LLM_PROVIDER setting.
  ARGS:
    prompt: a str prompt
  RETURNS:
    a str of the LLM's response.
  """
  if LLM_PROVIDER == "claude":
    return Claude_Opus_request(prompt)

  # Default to OpenAI GPT-4
  temp_sleep()

  try:
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]

  except:
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt):
  """
  Given a prompt, make a request to the configured LLM provider.
  Routes to either GPT-3.5-turbo or Claude 3.5 Sonnet based on LLM_PROVIDER setting.
  ARGS:
    prompt: a str prompt
  RETURNS:
    a str of the LLM's response.
  """
  if LLM_PROVIDER == "claude":
    return Claude_request(prompt)

  # Default to OpenAI GPT-3.5-turbo
  # temp_sleep()
  try:
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]

  except:
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  """
  Routes to either GPT-4 or Claude Opus based on LLM_PROVIDER setting.
  """
  if LLM_PROVIDER == "claude":
    return Claude_Opus_safe_generate_response(prompt, example_output, special_instruction,
                                               repeat, fail_safe_response, func_validate,
                                               func_clean_up, verbose)

  # Default to OpenAI GPT-4
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):

    try:
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass

  return False


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  """
  Routes to either GPT-3.5-turbo or Claude 3.5 Sonnet based on LLM_PROVIDER setting.
  """
  if LLM_PROVIDER == "claude":
    return Claude_safe_generate_response(prompt, example_output, special_instruction,
                                          repeat, fail_safe_response, func_validate,
                                          func_clean_up, verbose)

  # Default to OpenAI GPT-3.5-turbo
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):

    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass

  return False


def Claude_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  """
  Safe wrapper for Claude API with retry logic and JSON parsing.
  Uses Claude 3.5 Sonnet by default.
  """
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print("CLAUDE PROMPT")
    print(prompt)

  for i in range(repeat):

    try:
      curr_response = Claude_request(prompt).strip()
      end_index = curr_response.rfind('}') + 1
      curr_response = curr_response[:end_index]
      curr_response = json.loads(curr_response)["output"]

      if func_validate(curr_response, prompt=prompt):
        return func_clean_up(curr_response, prompt=prompt)

      if verbose:
        print("---- repeat count: \n", i, curr_response)
        print(curr_response)
        print("~~~~")

    except:
      pass

  return False


def Claude_Opus_safe_generate_response(prompt,
                                        example_output,
                                        special_instruction,
                                        repeat=3,
                                        fail_safe_response="error",
                                        func_validate=None,
                                        func_clean_up=None,
                                        verbose=False):
  """
  Safe wrapper for Claude Opus API with retry logic and JSON parsing.
  Uses Claude 3 Opus model.
  """
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print("CLAUDE OPUS PROMPT")
    print(prompt)

  for i in range(repeat):

    try:
      curr_response = Claude_Opus_request(prompt).strip()
      end_index = curr_response.rfind('}') + 1
      curr_response = curr_response[:end_index]
      curr_response = json.loads(curr_response)["output"]

      if func_validate(curr_response, prompt=prompt):
        return func_clean_up(curr_response, prompt=prompt)

      if verbose:
        print("---- repeat count: \n", i, curr_response)
        print(curr_response)
        print("~~~~")

    except:
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  try: 
    response = openai.Completion.create(
                model=gpt_parameter["engine"],
                prompt=prompt,
                temperature=gpt_parameter["temperature"],
                max_tokens=gpt_parameter["max_tokens"],
                top_p=gpt_parameter["top_p"],
                frequency_penalty=gpt_parameter["frequency_penalty"],
                presence_penalty=gpt_parameter["presence_penalty"],
                stream=gpt_parameter["stream"],
                stop=gpt_parameter["stop"],)
    return response.choices[0].text
  except: 
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  return openai.Embedding.create(
          input=[text], model=model)['data'][0]['embedding']


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)




















