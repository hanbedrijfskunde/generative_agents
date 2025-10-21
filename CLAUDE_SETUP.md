# Using Claude API with Generative Agents

This project now supports using Anthropic's Claude API as an alternative to OpenAI's GPT models for running the generative agents simulation.

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install both the OpenAI SDK (required for embeddings) and the Anthropic SDK (for Claude).

### 2. Configure API Keys

Create `reverie/backend_server/utils.py` with the following configuration:

```python
# API Configuration
# Choose your LLM backend: "openai" or "claude"
llm_provider = "claude"  # Set to "claude" to use Anthropic's Claude API

# OpenAI API Key (required for embeddings, even when using Claude)
openai_api_key = "sk-..."

# Anthropic API Key (required when using Claude)
anthropic_api_key = "sk-ant-..."

# Put your name
key_owner = "Your Name"

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose
debug = True
```

### 3. Get Your API Keys

**Anthropic API Key (for Claude):**
- Sign up at [console.anthropic.com](https://console.anthropic.com/)
- Navigate to API Keys section
- Create a new API key

**OpenAI API Key (for embeddings):**
- Sign up at [platform.openai.com](https://platform.openai.com/)
- Navigate to API Keys section
- Create a new API key

**Note:** Even when using Claude for the LLM, you still need an OpenAI API key for embeddings, as Claude doesn't provide an embeddings API.

## Configuration Options

### Use Claude (Anthropic)

```python
llm_provider = "claude"
openai_api_key = "sk-..."  # For embeddings only
anthropic_api_key = "sk-ant-..."  # For LLM
```

**Models used:**
- LLM: Claude 3.5 Sonnet (claude-3-5-sonnet-20241022)
- Advanced LLM: Claude 3 Opus (claude-3-opus-20240229)
- Embeddings: OpenAI text-embedding-ada-002

### Use OpenAI (Original)

```python
llm_provider = "openai"
openai_api_key = "sk-..."
# anthropic_api_key not needed
```

**Models used:**
- LLM: GPT-3.5-turbo
- Advanced LLM: GPT-4
- Embeddings: OpenAI text-embedding-ada-002

## How It Works

The integration uses a routing architecture:

1. **Transparent Routing:** All existing code remains unchanged. The `ChatGPT_request()` and `GPT4_request()` functions automatically route to the configured provider.

2. **Model Mapping:**
   - `ChatGPT_request()` → Claude 3.5 Sonnet (when `llm_provider = "claude"`)
   - `GPT4_request()` → Claude 3 Opus (when `llm_provider = "claude"`)

3. **Embeddings:** Always uses OpenAI's `text-embedding-ada-002` model regardless of LLM provider, as this is required for the memory retrieval system.

## Code Structure

The main changes are in:
- `reverie/backend_server/persona/prompt_template/gpt_structure.py` - API wrapper functions
- `reverie/backend_server/utils.py` - Configuration (user-created)
- `requirements.txt` - Added Anthropic SDK

## Running the Simulation

Follow the same steps as in the main README:

1. Start the environment server:
   ```bash
   cd environment/frontend_server
   python manage.py runserver
   ```

2. Start the simulation server:
   ```bash
   cd reverie/backend_server
   python reverie.py
   ```

The simulation will automatically use Claude if configured!

## Cost Considerations

**Claude Pricing (as of 2024):**
- Claude 3.5 Sonnet: $3/MTok input, $15/MTok output
- Claude 3 Opus: $15/MTok input, $75/MTok output

**OpenAI Pricing (for comparison):**
- GPT-3.5-turbo: $0.50/MTok input, $1.50/MTok output
- GPT-4: $30/MTok input, $60/MTok output
- Embeddings (ada-002): $0.10/MTok

**Notes:**
- Claude 3.5 Sonnet is comparable in cost to GPT-4
- You'll still incur minimal OpenAI costs for embeddings
- Running multi-agent simulations can be expensive with either provider
- Save your simulation frequently to avoid losing progress

## Troubleshooting

**Error: "CLAUDE ERROR: Anthropic client not initialized"**
- Check that `anthropic_api_key` is set in `utils.py`
- Verify the API key is valid
- Ensure `anthropic` package is installed: `pip install anthropic`

**Error: "ModuleNotFoundError: No module named 'anthropic'"**
- Run: `pip install -r requirements.txt`

**Embeddings errors even with Claude configured:**
- Verify your OpenAI API key is valid (required for embeddings)
- Check that `openai_api_key` is set in `utils.py`

## Benefits of Using Claude

1. **Performance:** Claude 3.5 Sonnet offers strong reasoning and instruction-following
2. **Context Window:** 200K token context window (vs 4K-8K for GPT-3.5)
3. **Alternative Provider:** Reduces dependency on a single API provider
4. **Cost Flexibility:** Choose the provider that fits your budget

## Switching Between Providers

You can easily switch between OpenAI and Claude by changing one line in `utils.py`:

```python
llm_provider = "openai"  # or "claude"
```

No other code changes needed!
