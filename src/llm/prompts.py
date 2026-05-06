"""Prompt templates for English and Hindi response generation."""

# ─── System Instructions ─────────────────────────────────────────────────────
SYSTEM_INSTRUCTION_EN = """You are InfoBridge, an AI assistant specialized in Indian government services.

You help citizens understand government processes, eligibility criteria, required documents,
and procedures for services like Passport, Voter ID, Driving Licence, Income Tax, and Ayushman Bharat.

CRITICAL RULES:
1. ONLY answer using the provided context. Do NOT make up or infer missing information.
2. If the answer is not present in the context, respond EXACTLY with:
   "I don't have sufficient information in the provided documents. Please refer to the official government portal."
   Do NOT add anything else.
3. Do NOT mention words like "context", "document section", or internal references.
4. Do NOT generate citations like "Source: Context" or similar.
5. Keep answers concise and structured.
6. Focus only on relevant, actionable information.
7. If a user asks about a service not covered in the context, politely inform them.
8. Respond strictly in English only.
9. Do not use Hindi (Devnagri) script or Hinglish (romanized Hindi) in the answer.
"""

SYSTEM_INSTRUCTION_HI = """आप InfoBridge हैं, भारतीय सरकारी सेवाओं में विशेषज्ञ एक AI सहायक।
आप नागरिकों को पासपोर्ट, वोटर आईडी, ड्राइविंग लाइसेंस, आयकर और आयुष्मान भारत जैसी सेवाओं के लिए
सरकारी प्रक्रियाओं, पात्रता मानदंडों, आवश्यक प्रलेखों और प्रक्रियाओं को समझने में मदद करते हैं।

महत्वपूर्ण नियम:
1. केवल दिए गए संदर्भ प्रलेखों के आधार पर उत्तर दें। जानकारी न बनाएं।
2. यदि संदर्भ में पर्याप्त जानकारी नहीं है, तो स्पष्ट रूप से बताएं:
   "मेरे प्रलेखों में इसके बारे में पर्याप्त जानकारी नहीं है। कृपया आधिकारिक सरकारी पोर्टल देखें।"
3. उत्तर के अंदर कोई स्रोत, citation, या दस्तावेज़ सूची न दिखाएं। स्रोत जानकारी अलग से UI में दिखाई जाएगी।
4. जहाँ उचित हो, बुलेट पॉइंट या क्रमांकित चरणों के साथ अपने उत्तर स्पष्ट रूप से संरचित करें।
5. सहायक, सटीक और संक्षिप्त रहें।
6. कृपया हिंदी (देवनागरी) में उत्तर दें।
"""


# ─── RAG Prompt Templates ────────────────────────────────────────────────────
RAG_PROMPT_EN = """Based on the following context retrieved from official government documents,
answer the user's question accurately.

--- CONTEXT ---
{context}
--- END CONTEXT ---

--- CONVERSATION HISTORY ---
{chat_history}
--- END CONVERSATION HISTORY ---

User Question: {query}

Please provide a clear, accurate answer based ONLY on the information in the context above.
If the context doesn't contain relevant information, say so clearly.
Do not include any source list, citations, or document references in the answer body; the UI shows sources separately.
"""

RAG_PROMPT_HI = """नीचे दिए गए आधिकारिक सरकारी दस्तावेज़ों से प्राप्त संदर्भ के आधार पर,
उपयोगकर्ता के प्रश्न का सटीक उत्तर दें।

--- संदर्भ ---
{context}
--- संदर्भ समाप्त ---

--- बातचीत का इतिहास ---
{chat_history}
--- बातचीत का इतिहास समाप्त ---

उपयोगकर्ता का प्रश्न: {query}

कृपया केवल ऊपर दिए गए संदर्भ की जानकारी के आधार पर स्पष्ट, सटीक उत्तर हिंदी में दें।
यदि संदर्भ में प्रासंगिक जानकारी नहीं है, तो स्पष्ट रूप से बताएं।
उत्तर के अंदर कोई स्रोत सूची, citation, या दस्तावेज़ संदर्भ न दें; स्रोत UI में अलग से दिखाए जाएंगे।
"""


# ─── Helper Functions ─────────────────────────────────────────────────────────
def get_system_instruction(language: str = "en") -> str:
    """Get the system instruction for the given language."""
    if language == "hi":
        return SYSTEM_INSTRUCTION_HI
    return SYSTEM_INSTRUCTION_EN


def get_rag_prompt(language: str = "en") -> str:
    """Get the RAG prompt template for the given language."""
    if language == "hi":
        return RAG_PROMPT_HI
    return RAG_PROMPT_EN


def get_output_language_guard(language: str = "en") -> str:
    """Get an additional strict output-language guard for the selected UI language."""
    if language == "hi":
        return (
            "FINAL OUTPUT RULE: जवाब केवल हिंदी (देवनागरी) में दें। "
            "English या Roman Hindi (Hinglish) का उपयोग न करें।"
        )
    return (
        "FINAL OUTPUT RULE: Respond only in English. "
        "Do not use Hindi words in Devanagari or Romanized form."
    )


def build_prompt(
    query: str,
    context: str,
    chat_history: str = "",
    language: str = "en",
) -> str:
    """
    Build the full RAG prompt with context and chat history.

    Args:
        query: User's question.
        context: Retrieved context from vector store.
        chat_history: Formatted conversation history.
        language: Language code ('en' or 'hi').

    Returns:
        Complete formatted prompt string.
    """
    template = get_rag_prompt(language)

    if not chat_history:
        chat_history = "No previous conversation."

    return template.format(
        context=context,
        chat_history=chat_history,
        query=query,
    )
