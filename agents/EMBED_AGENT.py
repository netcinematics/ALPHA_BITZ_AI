# 3.1 ) EMBEDDING_AGENT - get EMBEDDING DATA from GEMINI - with LOGGER.

print(f"\n ASK GEMINI for EMBED DATA... (takes a while!)")
import json 

try:
    # INIT GENAI client. NECESSARY for EMBED DATA LOAD customization.
    client = genai.Client()
    print("✅ genai client created.")
except Exception as e:
    # Placeholder for environment where client is not configured
    print(f"Warning: Could not initialize genai client. Ensure proper authentication: {e}")

# --- Embedding Model and Dimension Constants ---
EMBEDDING_MODEL = 'text-embedding-004' 
EMBEDDING_DIMENSIONS = 768
N_DIMENSIONS_REAL = EMBEDDING_DIMENSIONS


# --- AGENT AND TOOL DEFINITIONS ---
def get_word_embeddings(words: List[str]) -> str: 
    """
    Generates high-dimensional vector embeddings for a list of words.
    Returns a JSON STRING representing the list of lists.
    """
    print(f"🤖 Embedding Agent Tool: Requesting embeddings for {len(words)} words...")

    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=words,
        )        

        embedding_objects = response.embeddings
        # Get the raw values
        raw_list = [obj.values for obj in embedding_objects]
        
        print(f"✅ Embeddings generated. converting to JSON String...")
        
        # --- STRINGIFY THE DATA ---
        # We convert the massive list to a string so the LLM treats it as a "Text Blob"
        # instead of "1,500 numbers it has to think about".
        return json.dumps(raw_list)        

    except Exception as e:
        print(f"🛑 Unexpected Error: {e}")
        return "[]"

# _________________________EMBEDDING_AGENT____________

EMBEDDING_AGENT = LlmAgent(
    name="EmbeddingGeneratorAgent",
    model=Gemini(
        model=MODEL_NAME, 
        retry_options=retry_config),
    instruction="""
        You are a data retrieval tool.
        1. Call `get_word_embeddings`.
        2. The tool returns a JSON String.
        3. Your ONLY job is to output that exact JSON String.
        4. Do not parse it. Do not explain it.
        """,
    tools=[get_word_embeddings],
)

# --- 1. SETUP THE LOGGING RUNNER ---
EMBEDDING_RUNNER_LOG = InMemoryRunner(
    agent=EMBEDDING_AGENT,
    plugins=[LoggingPlugin()], # < ------------------------ UNCOMMENT TO TURN ON LOGGER (to DEBUG)
)

# --- HELPER FUNCTION ( ensure it catches the string) ---
def get_agent_text_or_tool_response(response_obj):
    if isinstance(response_obj, str): return response_obj
    
    if isinstance(response_obj, list):
        last_event = response_obj[-1]
        try:
            # 1. Check Function Response (If agent stopped early)
            if hasattr(last_event, 'parts'):
                 for part in last_event.parts:
                     if hasattr(part, 'function_response'):
                         return part.function_response.response.result # .result is key for JSON
            
            # 2. Check Text Content (If agent echoed the JSON)
            if hasattr(last_event, 'content') and last_event.content:
                # Dig deeper for Gemini 2.0 structure
                if hasattr(last_event.content, 'parts'):
                    return last_event.content.parts[0].text
                return last_event.content
        except:
            pass
            
    return str(response_obj)

# ----------------------------------------------------------
# --- EXECUTION ---
TEST_WORDS = ["words", "WORDZ"]
print(f"--- QUERY for EMBEDS of: {TEST_WORDS} ---")

# 2. RUN THE AGENT 
raw_response = await EMBEDDING_RUNNER_LOG.run_debug(
    f"Generate the vector embeddings for this: {TEST_WORDS}"
)

# 3. UNWRAP THE DATA 
embedding_result_string = get_agent_text_or_tool_response(raw_response)

# 4. RE-PARSE FOR USE (Since we returned a string, we load it back to a list)
try:
    final_vector_list = json.loads(embedding_result_string)
    print(f"\n✅ SUCCESS! Retrieved {len(final_vector_list)} vectors.")
    print(f"Vector Dimensions: {len(final_vector_list[0])}")
    print(f"First 5 floats of word 1: {final_vector_list[0][:5]}")
except Exception as e:
    print(f"⚠️ Parsing failed: {e}")
    print(f"Raw Output: {embedding_result_string[:100]}...")