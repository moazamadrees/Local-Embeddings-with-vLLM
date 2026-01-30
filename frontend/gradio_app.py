import gradio as gr
import requests
from typing import List, Tuple

API_URL = "http://localhost:8000"

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"✓ API Online | Documents: {data.get('documents_loaded', 'N/A')}"
        return "✗ API Offline"
    except:
        return "✗ API Offline - Cannot connect"

def get_stats():
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"""
**System Statistics:**
- Total Documents: {data.get('total_documents', 'N/A')}
- Embedding Model: {data.get('embedding_model', 'N/A').split('/')[-1]}
- LLM Model: {data.get('llm_model', 'N/A').split('/')[-1]}
"""
        return "Stats unavailable"
    except:
        return "Stats unavailable"

def chat(message: str, history: List[Tuple[str, str]], top_k: int) -> Tuple[List[Tuple[str, str]], str]:
    if not message.strip():
        return history, ""
    
    try:
        history_for_api = []
        for user_msg, bot_msg in history:
            history_for_api.append({"role": "user", "content": user_msg})
            history_for_api.append({"role": "assistant", "content": bot_msg})
        
        payload = {
            "message": message,
            "history": history_for_api,
            "top_k": top_k
        }
        
        response = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer generated")
            citations = data.get("citations", [])
            
            if citations:
                answer += f"\n\n**Citations ({len(citations)}):**\n"
                for i, citation in enumerate(citations[:3], 1):
                    answer += f"\n{i}. {citation[:200]}...\n"
            
            history.append((message, answer))
            return history, ""
        else:
            error_msg = f"Error: {response.status_code}"
            history.append((message, error_msg))
            return history, ""
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append((message, error_msg))
        return history, ""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), title="UET Department Chatbot") as demo:
    gr.Markdown("""
    # 🎓 UET Department Information Chatbot
    Ask questions about UET departments, programs, admissions, and faculty.
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
                show_label=True,
                container=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about UET departments...",
                    lines=2,
                    scale=4
                )
                submit = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear = gr.Button("Clear Chat")
            
            gr.Markdown("""
            **Example Questions:**
            - What are the admission requirements for Computer Science?
            - Tell me about the faculty in the Electrical Engineering department.
            - What programs are offered by the engineering departments?
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            status = gr.Textbox(
                label="API Status",
                value=check_api_health(),
                interactive=False
            )
            refresh_btn = gr.Button("Refresh Status")
            
            top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Context Chunks",
                info="Number of document chunks to retrieve"
            )
            
            stats = gr.Markdown(get_stats())
            refresh_stats_btn = gr.Button("Refresh Stats")
            
            gr.Markdown("""
            ### ℹ️ About
            This chatbot uses RAG (Retrieval-Augmented Generation) with:
            - Local embeddings (sentence-transformers)
            - ChromaDB vector storage
            - TinyLlama language model
            - Guardrail for scope validation
            
            **Note:** Only department-related questions will be answered.
            """)
    
    submit.click(chat, inputs=[msg, chatbot, top_k], outputs=[chatbot, msg])
    msg.submit(chat, inputs=[msg, chatbot, top_k], outputs=[chatbot, msg])
    clear.click(lambda: [], outputs=[chatbot])
    refresh_btn.click(check_api_health, outputs=[status])
    refresh_stats_btn.click(get_stats, outputs=[stats])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
