import gradio as gr
from document_processor import DocumentProcessor
from retriever_builder import RetrieverBuilder
from agent_workflow import AgentWorkflow

processor = DocumentProcessor()
retriever_builder = RetrieverBuilder()
workflow = AgentWorkflow()

def process_query(files, question):
    if not files:
        return "Please upload at least one document.", ""
    if not question.strip():
        return "Please enter a question.", ""
        
    try:
        chunks = processor.process(files)
        retriever = retriever_builder.build_hybrid_retriever(chunks)
        result = workflow.full_pipeline(question, retriever)
        return result["draft_answer"], result["verification_report"]
    except Exception as e:
        return f"Error: {str(e)}", ""

demo = gr.Interface(
    fn=process_query,
    inputs=[
        gr.Files(label="Upload Documents (PDF, TXT, MD)"),
        gr.Textbox(label="Ask a Question", placeholder="Enter your query here...")
    ],
    outputs=[
        gr.Textbox(label="DocChat Verified Answer"),
        gr.Textbox(label="Verification Report")
    ],
    title="DocChat: Multi-Agent RAG Document Intelligence",
    description="An advanced multi-agent RAG system featuring Docling parsing, hybrid retrieval (BM25 + ChromaDB), and LangGraph hallucination verification."
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=5000)
