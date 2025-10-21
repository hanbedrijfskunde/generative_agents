# Using Claude API with Generative Agents

This project now supports using Anthropic's Claude API as an alternative to OpenAI's GPT models, and multiple embedding providers including **free local embeddings**.

## Quick Setup (100% Free from OpenAI)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the Anthropic SDK (for Claude LLM) and Sentence Transformers (for free local embeddings).

### 2. Configure API Keys

Create `reverie/backend_server/utils.py` with the following configuration:

```python
# API Configuration
# Choose your LLM backend: "openai" or "claude"
llm_provider = "claude"  # Use Claude for LLM

# Choose your embedding backend: "openai", "voyage", "cohere", or "sentence-transformers"
embedding_provider = "sentence-transformers"  # Free local embeddings!

# API Keys (only needed for the providers you choose)
anthropic_api_key = "sk-ant-..."  # Required for Claude
# No OpenAI key needed with this setup!

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

**Anthropic API Key (for Claude LLM):**
- Sign up at [console.anthropic.com](https://console.anthropic.com/)
- Navigate to API Keys section
- Create a new API key
- You get $5 free credit to start!

**No other API keys needed** if you use the recommended setup above!

## Configuration Options

### LLM Providers

**Claude (Recommended):**
```python
llm_provider = "claude"
anthropic_api_key = "sk-ant-..."
```

Models used:
- Standard: Claude 3.5 Sonnet (claude-3-5-sonnet-20241022)
- Advanced: Claude 3 Opus (claude-3-opus-20240229)

**OpenAI (Original):**
```python
llm_provider = "openai"
openai_api_key = "sk-..."
```

Models used:
- Standard: GPT-3.5-turbo
- Advanced: GPT-4

### Embedding Providers

**Sentence Transformers (FREE, Recommended):**
```python
embedding_provider = "sentence-transformers"
# No API key needed - runs locally!
```

- Model: all-MiniLM-L6-v2 (default)
- Dimension: 384
- Cost: FREE
- Speed: Fast on CPU, very fast on GPU
- Quality: Excellent for semantic similarity
- Privacy: All data stays local

**Voyage AI (High Quality):**
```python
embedding_provider = "voyage"
voyage_api_key = "pa-..."
```

- Model: voyage-2 (default)
- Dimension: 1024
- Cost: $0.12/MTok
- Quality: Often outperforms OpenAI
- Get API key: [dash.voyageai.com](https://dash.voyageai.com/)

**Cohere (Good Alternative):**
```python
embedding_provider = "cohere"
cohere_api_key = "..."
```

- Model: embed-english-v3.0 (default)
- Dimension: 1024
- Cost: $0.10/MTok
- Quality: Competitive with OpenAI
- Get API key: [dashboard.cohere.com](https://dashboard.cohere.com/)

**OpenAI (Original):**
```python
embedding_provider = "openai"
openai_api_key = "sk-..."
```

- Model: text-embedding-ada-002
- Dimension: 1536
- Cost: $0.10/MTok

## How It Works

The integration uses a routing architecture:

1. **Transparent Routing:** All existing code remains unchanged. Functions automatically route to the configured provider.

2. **LLM Model Mapping:**
   - `ChatGPT_request()` → Claude 3.5 Sonnet (when `llm_provider = "claude"`)
   - `GPT4_request()` → Claude 3 Opus (when `llm_provider = "claude"`)

3. **Embedding Routing:**
   - `get_embedding()` → Routes to your configured embedding provider
   - Supports: Sentence Transformers (local), Voyage AI, Cohere, or OpenAI
   - Memory retrieval system works with any provider

## Code Structure

The main changes are in:
- `reverie/backend_server/persona/prompt_template/gpt_structure.py` - API wrapper functions for LLM and embeddings
- `reverie/backend_server/utils.py` - Configuration (user-created)
- `requirements.txt` - Added Anthropic SDK, Sentence Transformers, Voyage AI, and Cohere support

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

### LLM Pricing (as of 2024-2025)

**Claude:**
- Claude 3.5 Sonnet: $3/MTok input, $15/MTok output
- Claude 3 Opus: $15/MTok input, $75/MTok output

**OpenAI:**
- GPT-3.5-turbo: $0.50/MTok input, $1.50/MTok output
- GPT-4: $30/MTok input, $60/MTok output

### Embedding Pricing

**FREE Options:**
- **Sentence Transformers: $0** (runs locally, highly recommended!)

**Paid Options:**
- Voyage AI: $0.12/MTok
- Cohere: $0.10/MTok
- OpenAI: $0.10/MTok

### Recommended Budget-Friendly Setup

**100% Free from OpenAI:**
```python
llm_provider = "claude"
embedding_provider = "sentence-transformers"
```

**Completely Free (if you have local GPU):**
- Use local LLM via Ollama + Sentence Transformers
- See documentation for local LLM setup

**Notes:**
- Running multi-agent simulations can be expensive with API-based providers
- Sentence Transformers eliminates embedding costs entirely
- Save your simulation frequently to avoid losing progress
- Consider starting with a 3-agent simulation to test costs

## Troubleshooting

### LLM Errors

**Error: "CLAUDE ERROR: Anthropic client not initialized"**
- Check that `anthropic_api_key` is set in `utils.py`
- Verify the API key is valid at [console.anthropic.com](https://console.anthropic.com/)
- Ensure `anthropic` package is installed: `pip install anthropic`

**Error: "ModuleNotFoundError: No module named 'anthropic'"**
- Run: `pip install -r requirements.txt`

### Embedding Errors

**Error: "ModuleNotFoundError: No module named 'sentence_transformers'"**
- Run: `pip install -r requirements.txt`
- Or install directly: `pip install sentence-transformers`

**Error: "Voyage AI client not initialized"**
- Check that `voyage_api_key` is set in `utils.py`
- Get API key from [dash.voyageai.com](https://dash.voyageai.com/)
- Ensure `voyageai` package is installed: `pip install voyageai`

**Error: "Cohere client not initialized"**
- Check that `cohere_api_key` is set in `utils.py`
- Get API key from [dashboard.cohere.com](https://dashboard.cohere.com/)
- Ensure `cohere` package is installed: `pip install cohere`

**Sentence Transformers downloading model on first run:**
- This is normal! The model downloads once (~80MB) and caches locally
- Future runs will be instant
- Default model: all-MiniLM-L6-v2

**Slow embedding generation with Sentence Transformers:**
- First time: Model is downloading
- CPU mode: Normal, ~10-50ms per embedding
- To speed up: Use GPU if available (automatic detection)
- Consider using a smaller model if needed

## Benefits of This Setup

### LLM Benefits (Claude)

1. **Performance:** Claude 3.5 Sonnet offers strong reasoning and instruction-following
2. **Context Window:** 200K token context window (vs 4K-8K for GPT-3.5)
3. **Alternative Provider:** Reduces dependency on OpenAI
4. **Cost Competitive:** Similar pricing to GPT-4, better than GPT-4

### Embedding Benefits (Sentence Transformers)

1. **100% Free:** No API costs, no usage limits
2. **Privacy:** All data stays on your machine
3. **Fast:** GPU-accelerated if available, fast even on CPU
4. **No API Keys:** One less thing to manage
5. **Offline:** Works without internet connection
6. **Quality:** Excellent for semantic similarity tasks
7. **No Rate Limits:** Process as many embeddings as you want

### Combined Benefits

- **Zero OpenAI Dependency:** Run entirely without OpenAI
- **Cost Savings:** Only pay for Claude API calls, embeddings are free
- **Flexibility:** Mix and match any LLM + any embedding provider
- **Privacy:** Keep your simulation data private with local embeddings

## Switching Between Providers

You can easily switch providers by changing two lines in `utils.py`:

```python
llm_provider = "openai"  # or "claude"
embedding_provider = "sentence-transformers"  # or "openai", "voyage", "cohere"
```

No other code changes needed!

## Example Configurations

**Most Cost-Effective (Recommended):**
```python
llm_provider = "claude"
embedding_provider = "sentence-transformers"
anthropic_api_key = "sk-ant-..."
```

**Best Quality:**
```python
llm_provider = "claude"
embedding_provider = "voyage"
anthropic_api_key = "sk-ant-..."
voyage_api_key = "pa-..."
```

**Original OpenAI Setup:**
```python
llm_provider = "openai"
embedding_provider = "openai"
openai_api_key = "sk-..."
```

**Mixed (Claude + OpenAI embeddings):**
```python
llm_provider = "claude"
embedding_provider = "openai"
anthropic_api_key = "sk-ant-..."
openai_api_key = "sk-..."
```
