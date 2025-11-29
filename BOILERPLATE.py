# 1.1 ) AI APP BOILERPLATE (front-loading):
# _______________________________________________
print(" LOAD and INITIALIZE boilerplate tools...")

import os
from kaggle_secrets import UserSecretsClient

# DAY 1a, DAY 1b:
from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import AgentTool, FunctionTool, google_search
# DAY 2a:
from google.adk.runners import InMemoryRunner
from google.adk.code_executors import BuiltInCodeExecutor
# DAY 3a:
from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.sessions import DatabaseSessionService
from google.adk.tools.tool_context import ToolContext
# DAY 3b:
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.genai import types
# DAY 4:
from google.adk.plugins.logging_plugin import (LoggingPlugin,) 
import logging
# PYTHON:
from typing import List, Dict, Tuple, Any
import asyncio
import numpy as np
from google import genai
from google.genai.errors import APIError
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import ast
import re
import sys
import warnings
import json

print("✅ ADK components imported successfully.")
# client = genai.Client()
# print("✅ genai client created.")

# ________________________________________________________________________API_KEY:
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )
print("✅ Secret key initialized!")

# _________________________________________________________________ RETRY_CONFIG:
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)
print("✅ HttpRetry initialized.")


# _________________________________________________________________ APP CONSTANTS:
APP_NAME = "ALPHA_BITZ_CAPSTONE"  # Application
SESSION_ID = "MAIN_SESSION"  # Session
USER_ID = "spaceOTTER" #"default"  # User
# MODEL_NAME = "gemini-2.5-flash"
MODEL_NAME = "gemini-2.5-flash-lite"
print("✅ Constants Initialized.")
DATA_LOADED = False
print("✅ Data needs to be loaded.")
# _______________________________________________________________ MEMORY & SESSION:

# ADK's built-in Memory Service for development and testing
memory_service = ( InMemoryMemoryService() )

# InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()

#_________________________________________________________________ AUTO SAVE CALLBACK:

async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )
    print("Auto Saved.")


print("✅ Callback created.")

# ________________________________________________________________ CUSTOM RUN_SESSION:

async def run_session( # OPTIMIZED!
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
    # 🛑 NEW ARGUMENT: Allows suppression of console output # OPTIMIZED!
    suppress_print: bool = False,    
):

    # Get app name from the Runner
    app_name = runner_instance.app_name
    print("______________________________") # OPTIMIZED.
    print(f"### APP: {app_name}")
    print(f"### SESSION: {session_name}")
    print("______________________________")

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    final_response_text = "" # Initialize variable to capture the output

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    event_text = event.content.parts[0].text # Capture the text
                    
                    # Filter out empty or "None" responses before printing
                    if (
                        event_text != "None"
                        and event_text
                    ):
                        # 🛑 Only print if suppress_print is False # OPTIMIZED!
                        if not suppress_print:
                            print(f"### AI MODEL: {MODEL_NAME} > ", event_text)
                            
                        final_response_text = event_text                        
                        # print(f"### AI MODEL: {MODEL_NAME} > ", event_text)
                        # final_response_text = event_text # Capture the last (final) text event
    else:
        print("No queries!")

    return final_response_text # ⬅️ RETURNS the final captured text. # OPTIMIZED!

print("✅ Helper functions initialized.")

# __________________________________________________________________ LOGGER: 
# Configure the basic logging settings
logging.basicConfig(
    level=logging.DEBUG, # Set the minimum level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    # format='%(asctime)s - %(levelname)s - %(message)s',
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
    # Direct output to stdout (the console stream Kaggle captures)
    stream=sys.stdout 
)

# Get a logger instance
logger = logging.getLogger(__name__)

logger.info("Starting ALPHABITZ notebook execution.")


print("✅ Logging configured")

print("✅ END BOILERPLATE.")
