"""Prompt builder for RAG with citation support."""

from __future__ import annotations
from typing import List, Dict, Any


class PromptBuilder:
    """
    Builds prompts for RAG (Retrieval-Augmented Generation).
    
    Handles context injection, citation formatting, and instruction templates.
    """
    
    # Default RAG prompt template
    DEFAULT_TEMPLATE = """You are a helpful assistant answering questions about scikit-learn, a machine learning library in Python.

Use the following context from the scikit-learn documentation to answer the question. If the answer is not in the context, say so.

When you reference information from the context, cite the source using [1], [2], etc. corresponding to the context chunks below.

Context:
{context}

Question: {query}

Answer (with citations):"""

    # Alternative template for more structured answers
    STRUCTURED_TEMPLATE = """You are an expert on scikit-learn answering technical questions.

Below is relevant documentation to help answer the question. Use this information to provide an accurate, detailed answer.

Guidelines:
- Answer the question directly and concisely
- Include code examples if relevant
- Cite sources using [1], [2], etc. for each piece of information
- If unsure or information is not in the context, say so clearly

Context from scikit-learn documentation:
{context}

Question: {query}

Provide a clear, well-cited answer:"""

    def __init__(self, template: str = None):
        """
        Initialize prompt builder.
        
        Args:
            template: Custom prompt template (uses DEFAULT_TEMPLATE if None)
        """
        self.template = template or self.DEFAULT_TEMPLATE
    
    def build_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_context_length: int = 4000,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Build a RAG prompt from query and retrieved chunks.
        
        Args:
            query: User's question
            context_chunks: List of retrieved chunk dictionaries
            max_context_length: Maximum characters for context
            include_metadata: Whether to include doc_id in citations
            
        Returns:
            Dictionary with 'prompt', 'context_chunks_used', and 'citations_map'
        """
        # Format context with citations
        context_parts = []
        citations_map = {}
        total_length = 0
        chunks_used = []
        
        for i, chunk in enumerate(context_chunks, start=1):
            # Format context chunk with citation number
            content = chunk.get("content", "")
            doc_id = chunk.get("doc_id", "unknown")
            chunk_id = chunk.get("chunk_id", "unknown")
            
            # Create citation entry
            if include_metadata:
                context_entry = f"[{i}] (Source: {doc_id})\n{content}"
            else:
                context_entry = f"[{i}] {content}"
            
            # Check if adding this chunk would exceed max length
            if total_length + len(context_entry) > max_context_length:
                # Stop adding more chunks
                break
            
            context_parts.append(context_entry)
            total_length += len(context_entry)
            chunks_used.append(chunk)
            
            # Store citation mapping
            citations_map[i] = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "content": content,
                "similarity_score": chunk.get("similarity_score", 0.0),
                "strategy": chunk.get("strategy", "unknown")
            }
        
        # Join context parts
        context_text = "\n\n".join(context_parts)
        
        # Build final prompt
        prompt = self.template.format(
            context=context_text,
            query=query
        )
        
        return {
            "prompt": prompt,
            "context_chunks_used": chunks_used,
            "citations_map": citations_map,
            "num_chunks_used": len(chunks_used),
            "context_length": total_length
        }
    
    def extract_citations(self, generated_text: str) -> List[int]:
        """
        Extract citation numbers from generated text.
        
        Args:
            generated_text: Text with citations like [1], [2]
            
        Returns:
            List of citation numbers found in the text
        """
        import re
        
        # Find all [N] patterns where N is a number
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, generated_text)
        
        # Convert to integers and remove duplicates while preserving order
        citations = []
        seen = set()
        for match in matches:
            num = int(match)
            if num not in seen:
                citations.append(num)
                seen.add(num)
        
        return citations
    
    def format_answer_with_sources(
        self,
        generated_text: str,
        citations_map: Dict[int, Dict[str, Any]],
        include_full_sources: bool = True
    ) -> str:
        """
        Format the generated answer with a sources section.
        
        Args:
            generated_text: Generated answer text
            citations_map: Mapping of citation numbers to chunk info
            include_full_sources: Whether to include detailed sources section
            
        Returns:
            Formatted text with answer and sources
        """
        if not include_full_sources:
            return generated_text
        
        # Extract citations used in the answer
        used_citations = self.extract_citations(generated_text)
        
        if not used_citations:
            return generated_text
        
        # Build sources section
        sources_lines = ["\n\nSources:"]
        for citation_num in used_citations:
            if citation_num in citations_map:
                info = citations_map[citation_num]
                doc_id = info["doc_id"]
                strategy = info.get("strategy", "unknown")
                score = info.get("similarity_score", 0.0)
                
                sources_lines.append(
                    f"[{citation_num}] {doc_id} "
                    f"(strategy: {strategy}, relevance: {score:.2f})"
                )
        
        sources_text = "\n".join(sources_lines)
        
        return generated_text + sources_text
    
    @staticmethod
    def get_available_templates() -> Dict[str, str]:
        """
        Get available prompt templates.
        
        Returns:
            Dictionary of template names to template strings
        """
        return {
            "default": PromptBuilder.DEFAULT_TEMPLATE,
            "structured": PromptBuilder.STRUCTURED_TEMPLATE,
        }

