# 1.4 ) OBSERVABILITY ----------------------------------------------

#  ____ SEARCH MEMORY SERVICE: _______________________________
print("🔍 SEARCH MEMORY SERVICE:")
search_memory_results = await memory_service.search_memory(
    app_name=APP_NAME, user_id=USER_ID, query="ALPHABITZ"
)

print(f"  Found {len(search_memory_results.memories)} relevant memories")
for memory in search_memory_results.memories:
    if memory.content and memory.content.parts:
        text = memory.content.parts[0].text[:80]
        print(f"  [{memory.author}]: {text}...")

# ______ SEARCH SESSION SERVICE: _______________
print("🔍 SEARCH SESSION SERVICE:")
session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
)
print("SESSION STATE:")
print(session.state)

# Let's see what's in the session
print("📝 SESSION CONTAINS:")
for event in session.events:
    if event.content and event.content.parts:
        # Initialize a list to hold all text parts for this event
        text_parts = []
        
        # Iterate through all parts in the event
        for part in event.content.parts:
            # Check if the part has a 'text' attribute (or key, depending on structure)
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)
            # Add logic here for other types if needed (e.g., images, function calls)

        # Print the role and the joined text content
        if text_parts:
            # Join all text parts into a single string for clean output
            print(f"  {event.content.role}: {' '.join(text_parts)}...")
