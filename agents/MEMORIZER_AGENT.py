# 1.2 ) MEMORIZER_AGENT with GITHUB_TOOL - to load ALPHABITZ GITHUB_DATA:
#       - Load ENHANCED LANGUAGE SYNTAX markdown from GITHUB.
# _______________________________________________________________________

async def get_GITHUB_DATA(): # ---------------------- CUSTOM MCP TOOL.
    print("AGENT is loading GITHUB DATA.")
    try:
        %cd /kaggle/working
        !git clone https://github.com/netcinematics/ALPHABITZ_AI.git
        %cd ALPHABITZ_AI
    except:
        print("Data is already loaded.")       
    
    file_path = "ALPHABITZ_SYNTAX_MIN_001.md"
    
    # 2. Use a 'with' block to open and read the file safely.
    #    'r' stands for read mode. 'utf-8' is the standard encoding for text files.
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 3. Read the entire content of the file into a variable
            markdown_content = f.read()
            print("Data loaded, saving to session.")
            resp = await run_session( MEMORIZER_AGENT_RUNNER,
                f"Read this markdown file, and learn a new language called ALPHABITZ: \n\n{markdown_content}",
                SESSION_ID,
            )

            print("✅ Session added to memory!")
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found in the current directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")    

# ____________________________________________________________ MEMORIZER LLMAGENT
MEMORIZER_AGENT = LlmAgent(
    name="MemorizerAgent",
    model=Gemini(
        model=MODEL_NAME,
        retry_options=retry_config
    ),
    instruction="""You are a smart data retriever agent. 
        To load GitHub Data, use the `get_GITHUB_DATA()` tool.
        Then read the variable markdown_content and internalize it.  
        Do not summarize the contents. Respond that the "data is loaded", after it is loaded.
        For answers about the data, be brief and concise. Do not explain, just answer succinctly.
        When referencing words with underscore syntax, always show the raw text.
        If you are asked to return JSON, be sure to use double quotes for keys.
        For example, render actual_acts_of_mentality, not actual\_acts\_of\_mentality.
        Also do not say:'I have already provided this information. Please refer to my previous response.'
        Instead, return the data if asked multiple times.""",
    tools=[get_GITHUB_DATA, load_memory], # -------- Agent access to MEMORY!
    after_agent_callback=auto_save_to_memory,  # --------------- AUTO SAVE!
)

# Step 3: Create the Runner
MEMORIZER_AGENT_RUNNER = Runner(
    agent=MEMORIZER_AGENT, 
    app_name=APP_NAME, 
    session_service=session_service,
    memory_service=memory_service,  # ------ Memory service available!
)
print("✅ MEMORIZER Runner created, with SESSION and MEMORY service.")

# Use a prompt that clearly indicates the agent should perform the data load task.
data_load = await run_session( MEMORIZER_AGENT_RUNNER,
    "Load the GitHub data now by running the get_GITHUB_DATA() tool.",
    SESSION_ID,
)

# The after_agent_callback will automatically save this to memory.
auto_memory_test = await run_session( MEMORIZER_AGENT_RUNNER,
    "In ALPHABITZ language, AXI means actual_extra_intelligence.", # ---- PRIMING CONTEXT for TEST.
    SESSION_ID,
)

DATA_LOADED = True
print("✅ DATA is LOADED")
