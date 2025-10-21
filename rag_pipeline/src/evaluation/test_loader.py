"""Test set loader and validator for evaluation framework."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from ..utils.logger import get_logger


class TestQuestion:
    """Data class representing a test question."""
    
    def __init__(
        self,
        id: str,
        question: str,
        expected_topics: List[str],
        relevant_doc_ids: List[str],
        difficulty: str,
        category: str
    ):
        self.id = id
        self.question = question
        self.expected_topics = expected_topics
        self.relevant_doc_ids = relevant_doc_ids
        self.difficulty = difficulty
        self.category = category
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "expected_topics": self.expected_topics,
            "relevant_doc_ids": self.relevant_doc_ids,
            "difficulty": self.difficulty,
            "category": self.category
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestQuestion:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            question=data["question"],
            expected_topics=data["expected_topics"],
            relevant_doc_ids=data["relevant_doc_ids"],
            difficulty=data["difficulty"],
            category=data["category"]
        )


class TestLoader:
    """Loads and validates test sets for RAG evaluation."""
    
    def __init__(self, config, logger_name: str = "test_loader"):
        """
        Initialize test loader.
        
        Args:
            config: Configuration object
            logger_name: Logger name
        """
        self.config = config
        self.logger = get_logger(logger_name)
        
        # Get test set path from config
        test_set_path = config.get("evaluation.test_set_path", "data/evaluation/test_set.json")
        self.test_set_path = config.base_path / test_set_path
        
        self.logger.info(f"Test loader initialized with path: {self.test_set_path}")
    
    def load_test_set(self, path: Optional[Path] = None) -> List[TestQuestion]:
        """
        Load test set from JSON file.
        
        Args:
            path: Optional path to test set (uses default if None)
            
        Returns:
            List of TestQuestion objects
            
        Raises:
            FileNotFoundError: If test set file doesn't exist
            ValueError: If test set is invalid
        """
        path = path or self.test_set_path
        
        if not path.exists():
            raise FileNotFoundError(f"Test set not found: {path}")
        
        self.logger.info(f"Loading test set from: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        if "questions" not in data:
            raise ValueError("Test set must contain 'questions' field")
        
        # Parse questions
        questions = []
        for q_data in data["questions"]:
            try:
                question = TestQuestion.from_dict(q_data)
                questions.append(question)
            except KeyError as e:
                self.logger.warning(f"Skipping invalid question {q_data.get('id', 'unknown')}: missing field {e}")
                continue
        
        self.logger.info(f"Loaded {len(questions)} questions from test set")
        
        # Log statistics
        self._log_statistics(questions)
        
        return questions
    
    def _log_statistics(self, questions: List[TestQuestion]) -> None:
        """Log statistics about the test set."""
        
        # Count by category
        categories = {}
        for q in questions:
            categories[q.category] = categories.get(q.category, 0) + 1
        
        # Count by difficulty
        difficulties = {}
        for q in questions:
            difficulties[q.difficulty] = difficulties.get(q.difficulty, 0) + 1
        
        self.logger.info("Test set statistics:")
        self.logger.info(f"  Total questions: {len(questions)}")
        self.logger.info(f"  By category: {categories}")
        self.logger.info(f"  By difficulty: {difficulties}")
    
    def filter_by_category(
        self,
        questions: List[TestQuestion],
        category: str
    ) -> List[TestQuestion]:
        """
        Filter questions by category.
        
        Args:
            questions: List of questions
            category: Category to filter by
            
        Returns:
            Filtered list of questions
        """
        filtered = [q for q in questions if q.category == category]
        self.logger.info(f"Filtered to {len(filtered)} questions in category '{category}'")
        return filtered
    
    def filter_by_difficulty(
        self,
        questions: List[TestQuestion],
        difficulty: str
    ) -> List[TestQuestion]:
        """
        Filter questions by difficulty.
        
        Args:
            questions: List of questions
            difficulty: Difficulty to filter by
            
        Returns:
            Filtered list of questions
        """
        filtered = [q for q in questions if q.difficulty == difficulty]
        self.logger.info(f"Filtered to {len(filtered)} questions with difficulty '{difficulty}'")
        return filtered
    
    def get_question_by_id(
        self,
        questions: List[TestQuestion],
        question_id: str
    ) -> Optional[TestQuestion]:
        """
        Get a specific question by ID.
        
        Args:
            questions: List of questions
            question_id: Question ID
            
        Returns:
            TestQuestion or None if not found
        """
        for q in questions:
            if q.id == question_id:
                return q
        return None
    
    def validate_test_set(self, questions: List[TestQuestion]) -> bool:
        """
        Validate test set for completeness and consistency.
        
        Args:
            questions: List of questions to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not questions:
            self.logger.error("Test set is empty")
            return False
        
        # Check for duplicate IDs
        ids = [q.id for q in questions]
        if len(ids) != len(set(ids)):
            self.logger.error("Test set contains duplicate question IDs")
            return False
        
        # Check all questions have required fields
        for q in questions:
            if not q.question or not q.question.strip():
                self.logger.error(f"Question {q.id} has empty question text")
                return False
            
            if not q.expected_topics:
                self.logger.warning(f"Question {q.id} has no expected topics")
            
            if not q.relevant_doc_ids:
                self.logger.warning(f"Question {q.id} has no relevant doc IDs")
        
        self.logger.info("Test set validation passed")
        return True

