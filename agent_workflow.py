from typing import TypedDict, List
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever

class AgentWorkflow:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.compiled_workflow = self.build_workflow()

    def build_workflow(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)
        
        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            lambda state: "relevant" if state["is_relevant"] else "irrelevant",
            {"relevant": "research", "irrelevant": END}
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            lambda state: "re_research" if "Supported: NO" in state["verification_report"] else "end",
            {"re_research": "research", "end": END}
        )
        return workflow.compile()

    def _check_relevance_step(self, state: AgentState) -> dict:
        retriever = state["retriever"]
        docs = retriever.invoke(state["question"])
        if not docs:
            return {"is_relevant": False, "draft_answer": "Query is out of scope for the uploaded documents."}
        return {"is_relevant": True, "documents": docs}

    def _research_step(self, state: AgentState) -> dict:
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        prompt = f"Answer the question factually using only this context:\n\nContext:\n{context}\n\nQuestion: {state['question']}"
        response = self.llm.invoke(prompt)
        return {"draft_answer": response.content}

    def _verification_step(self, state: AgentState) -> dict:
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        prompt = f"Verify this answer against the context.\nFormat:\nSupported: YES/NO\nUnsupported Claims: None\nContradictions: None\nRelevant: YES/NO\n\nAnswer: {state['draft_answer']}\nContext:\n{context}"
        response = self.llm.invoke(prompt)
        return {"verification_report": response.content}

    def full_pipeline(self, question: str, retriever: EnsembleRetriever):
        initial_state = AgentState(
            question=question,
            documents=[],
            draft_answer="",
            verification_report="",
            is_relevant=False,
            retriever=retriever
        )
        final_state = self.compiled_workflow.invoke(initial_state)
        return {
            "draft_answer": final_state.get("draft_answer", "No answer generated."),
            "verification_report": final_state.get("verification_report", "No verification report.")
        }
