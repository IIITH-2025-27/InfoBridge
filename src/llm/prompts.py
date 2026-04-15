"""
Prompt Templates Module
Contains all prompt templates for the RAG pipeline, supporting English and Hindi.
"""

# ─── System Instructions ─────────────────────────────────────────────────────

SYSTEM_INSTRUCTION_EN = """You are InfoBridge, an AI assistant specialized in Indian government services. 
You help citizens understand government processes, eligibility criteria, required documents, 
and procedures for services like Passport, Voter ID, Driving Licence, Income Tax, and Ayushman Bharat.

CRITICAL RULES:
1. ONLY answer based on the provided context documents. Do NOT make up information.
2. If the context does not contain enough information to answer, clearly state: 
   "I don't have sufficient information about this in my documents. Please refer to the official government portal."
3. Always cite which source document(s) your answer comes from.
4. Structure your answers clearly with bullet points or numbered steps when appropriate.
5. Be helpful, accurate, and concise.
6. If a user asks about a service not covered in the context, politely inform them.
"""

SYSTEM_INSTRUCTION_HI = """आप InfoBridge हैं, भारतीय सरकारी सेवाओं में विशेषज्ञ एक AI सहायक।
आप नागरिकों को पासपोर्ट, वोटर आईडी, ड्राइविंग लाइसेंस, आयकर और आयुष्मान भारत जैसी सेवाओं के लिए 
सरकारी प्रक्रियाओं, पात्रता मानदंडों, आवश्यक दस्तावेज़ों और प्रक्रियाओं को समझने में मदद करते हैं।

महत्वपूर्ण नियम:
1. केवल दिए गए संदर्भ दस्तावेज़ों के आधार पर उत्तर दें। जानकारी न बनाएं।
2. यदि संदर्भ में पर्याप्त जानकारी नहीं है, तो स्पष्ट रूप से बताएं:
   "मेरे दस्तावेज़ों में इसके बारे में पर्याप्त जानकारी नहीं है। कृपया आधिकारिक सरकारी पोर्टल देखें।"
3. हमेशा बताएं कि आपका उत्तर किस स्रोत दस्तावेज़ से आया है।
4. जहाँ उचित हो, बुलेट पॉइंट या क्रमांकित चरणों के साथ अपने उत्तर स्पष्ट रूप से संरचित करें।
5. सहायक, सटीक और संक्षिप्त रहें।
6. कृपया हिंदी में उत्तर दें।
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
Cite the source documents in your answer.
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
अपने उत्तर में स्रोत दस्तावेज़ों का उल्लेख करें।
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
