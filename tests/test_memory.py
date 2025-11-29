# 1.3 ) TEST: DATA LOAD, SESSION & MEMORY _________________________

# ______ TEST SESSION _______________________________________
test_SESSION = await run_session( MEMORIZER_AGENT_RUNNER, [
        "What does aMENTZa mean?", # PROVES: GITHUB DATA LOAD, AGENT MEMORY, and vocabulary lookup.
        "From the syntax definition, what does aENaMENTZa mean?",  # "actual_acts_of_encouraging_mentality"
        "It is not defined in ALPHABITZ syntax, but try to inference the opposite of aENaMENTZa?" 
        # PROOF of AGENT INFERENCE! 
        # ANSWER: "aDISaMENTZa" or "actual_acts_of_discouraging_mentality"!
    ], SESSION_ID,
)

# ______ TEST AUTOSAVE MEMORY _______________________________
test_MEMORY = await run_session( MEMORIZER_AGENT_RUNNER,
    "What does AXI mean?", 
    # PROOF: of teaching Gemini a new language.
    # ANSWER: "Actual_Extra_Intelligence"
)
assert "actual_extra_intelligence" in test_MEMORY.lower(), "Memory test failed: AXI not found."
print("✅ TEST MEMORY passed: AXI found in memory.")