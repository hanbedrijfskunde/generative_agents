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

# Import embedding providers
voyage_client = None
cohere_client = None
sentence_transformer_model = None

try:
    import voyageai
    voyage_client = voyageai.Client(api_key=voyage_api_key) if 'voyage_api_key' in dir() and voyage_api_key else None
except ImportError:
    pass

try:
    import cohere
    cohere_client = cohere.Client(api_key=cohere_api_key) if 'cohere_api_key' in dir() and cohere_api_key else None
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    # Lazy load - only initialize if needed
    sentence_transformer_model = None
except ImportError:
    pass

# Get provider settings from utils (default to openai if not set)
LLM_PROVIDER = llm_provider if 'llm_provider' in dir() else "openai"
EMBEDDING_PROVIDER = embedding_provider if 'embedding_provider' in dir() else "openai"

# Get Claude model selection (default to sonnet if not set)
CLAUDE_MODEL = claude_model if 'claude_model' in dir() else "sonnet"

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

def get_claude_model_name(model_type="sonnet"):
  """
  Get the Claude model name based on model type.
  Args:
    model_type: "haiku", "sonnet", or "opus"
  Returns:
    Full model name string
  """
  models = {
    "haiku": "claude-3-5-haiku-20241022",
    "sonnet": "claude-3-5-sonnet-20241022",
    "opus": "claude-3-opus-20240229"
  }
  return models.get(model_type, "claude-3-5-sonnet-20241022")


def Claude_request(prompt, model=None):
  """
  Given a prompt, make a request to Anthropic Claude API and return the response.
  Uses the model specified in CLAUDE_MODEL config (default: sonnet).
  ARGS:
    prompt: a str prompt
    model: Optional model override ("haiku", "sonnet", or "opus")
  RETURNS:
    a str of Claude's response.
  """
  temp_sleep()

  try:
    if anthropic_client is None:
      return "CLAUDE ERROR: Anthropic client not initialized"

    model_name = get_claude_model_name(model or CLAUDE_MODEL)

    message = anthropic_client.messages.create(
      model=model_name,
      max_tokens=2000,
      messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

  except Exception as e:
    print(f"CLAUDE ERROR: {str(e)}")
    return "CLAUDE ERROR"


def Claude_Haiku_request(prompt):
  """
  Given a prompt, make a request to Anthropic Claude API using Haiku model.
  Haiku is the fastest and most cost-effective Claude model.
  ARGS:
    prompt: a str prompt
  RETURNS:
    a str of Claude's response.
  """
  return Claude_request(prompt, model="haiku")


def Claude_Opus_request(prompt):
  """
  Given a prompt, make a request to Anthropic Claude API using Opus model.
  Opus is the most capable Claude model.
  ARGS:
    prompt: a str prompt
  RETURNS:
    a str of Claude's response.
  """
  return Claude_request(prompt, model="opus")


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
  Uses the model specified in CLAUDE_MODEL config (default: sonnet).
  """
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print(f"CLAUDE PROMPT (using {CLAUDE_MODEL})")
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


def Claude_Haiku_safe_generate_response(prompt,
                                        example_output,
                                        special_instruction,
                                        repeat=3,
                                        fail_safe_response="error",
                                        func_validate=None,
                                        func_clean_up=None,
                                        verbose=False):
  """
  Safe wrapper for Claude Haiku API with retry logic and JSON parsing.
  Uses Claude 3.5 Haiku model (fastest and most cost-effective).
  """
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print("CLAUDE HAIKU PROMPT")
    print(prompt)

  for i in range(repeat):

    try:
      curr_response = Claude_Haiku_request(prompt).strip()
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
  Uses Claude 3 Opus model (most capable).
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


# ============================================================================
# #####################[SECTION 3: EMBEDDING FUNCTIONS] ######################
# ============================================================================

def get_openai_embedding(text, model="text-embedding-ada-002"):
  """Get embeddings from OpenAI API."""
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"
  return openai.Embedding.create(
          input=[text], model=model)['data'][0]['embedding']


def get_voyage_embedding(text, model="voyage-2"):
  """Get embeddings from Voyage AI API."""
  global voyage_client
  if voyage_client is None:
    raise ValueError("Voyage AI client not initialized. Check your voyage_api_key in utils.py")

  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"

  result = voyage_client.embed([text], model=model)
  return result.embeddings[0]


def get_cohere_embedding(text, model="embed-english-v3.0"):
  """Get embeddings from Cohere API."""
  global cohere_client
  if cohere_client is None:
    raise ValueError("Cohere client not initialized. Check your cohere_api_key in utils.py")

  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"

  result = cohere_client.embed(
    texts=[text],
    model=model,
    input_type="search_document"  # For storing in vector DB
  )
  return result.embeddings[0]


def get_sentence_transformer_embedding(text, model="all-MiniLM-L6-v2"):
  """Get embeddings from local Sentence Transformers model."""
  global sentence_transformer_model

  # Lazy load the model on first use
  if sentence_transformer_model is None:
    try:
      from sentence_transformers import SentenceTransformer
      sentence_transformer_model = SentenceTransformer(model)
    except Exception as e:
      raise ValueError(f"Failed to load Sentence Transformer model: {e}")

  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"

  # Returns numpy array, convert to list for consistency
  embedding = sentence_transformer_model.encode(text)
  return embedding.tolist()


def get_embedding(text, model=None):
  """
  Get embeddings using the configured provider.
  Routes to appropriate embedding provider based on EMBEDDING_PROVIDER setting.

  Args:
    text: Text to embed
    model: Optional model override. If not provided, uses default for provider.

  Returns:
    List of floats representing the embedding vector
  """
  if EMBEDDING_PROVIDER == "voyage":
    return get_voyage_embedding(text, model or "voyage-2")

  elif EMBEDDING_PROVIDER == "cohere":
    return get_cohere_embedding(text, model or "embed-english-v3.0")

  elif EMBEDDING_PROVIDER == "sentence-transformers":
    return get_sentence_transformer_embedding(text, model or "all-MiniLM-L6-v2")

  else:  # Default to OpenAI
    return get_openai_embedding(text, model or "text-embedding-ada-002")


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




















