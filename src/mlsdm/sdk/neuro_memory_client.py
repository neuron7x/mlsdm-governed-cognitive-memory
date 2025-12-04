"""
MLSDM Neuro Memory Client: Extended SDK with Memory and Decision APIs.

This module provides a unified client for:
- Memory operations (append, query)
- Decision-making with governance
- Agent step protocol
- Both local (direct) and remote (HTTP) modes

Usage:
    >>> from mlsdm.sdk import NeuroMemoryClient
    >>> 
    >>> # Local mode (no HTTP, uses engine directly)
    >>> client = NeuroMemoryClient(mode="local")
    >>> 
    >>> # Remote mode (HTTP API)
    >>> client = NeuroMemoryClient(mode="remote", base_url="http://localhost:8000")
    >>> 
    >>> # Memory operations
    >>> client.append_memory("Important fact about the user")
    >>> results = client.query_memory("What do I know about the user?")
    >>> 
    >>> # Decision-making
    >>> decision = client.decide("Should I proceed with this action?", risk_level="high")
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import requests

logger = logging.getLogger(__name__)


# ============================================================
# Data Transfer Objects
# ============================================================


@dataclass
class MemoryItem:
    """A single memory item from query results."""
    
    content: str
    similarity: float
    phase: float
    metadata: dict[str, Any] | None = None


@dataclass
class MemoryAppendResult:
    """Result from memory append operation."""
    
    success: bool
    memory_id: str | None
    phase: str
    accepted: bool
    memory_stats: dict[str, Any] | None = None
    message: str | None = None


@dataclass
class MemoryQueryResult:
    """Result from memory query operation."""
    
    success: bool
    results: list[MemoryItem] = field(default_factory=list)
    query_phase: str = "unknown"
    total_results: int = 0
    message: str | None = None


@dataclass
class ContourDecision:
    """Decision from a governance contour."""
    
    contour: str
    passed: bool
    score: float | None = None
    threshold: float | None = None
    notes: str | None = None


@dataclass
class DecideResult:
    """Result from decision operation."""
    
    response: str
    accepted: bool
    phase: str
    contour_decisions: list[ContourDecision] = field(default_factory=list)
    memory_context_used: int = 0
    risk_assessment: dict[str, Any] | None = None
    timing: dict[str, float] | None = None
    decision_id: str | None = None
    message: str | None = None


@dataclass
class AgentAction:
    """Agent action to take."""
    
    action_type: Literal["respond", "tool_call", "wait", "terminate"]
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class AgentStepResult:
    """Result from agent step operation."""
    
    action: AgentAction
    response: str
    phase: str
    accepted: bool
    updated_state: dict[str, Any] | None = None
    memory_updated: bool = False
    memory_context_used: int = 0
    step_id: str | None = None
    timing: dict[str, float] | None = None
    message: str | None = None


# ============================================================
# Exceptions
# ============================================================


class NeuroMemoryError(Exception):
    """Base exception for NeuroMemoryClient errors."""
    pass


class NeuroMemoryConnectionError(NeuroMemoryError):
    """Connection error when communicating with remote service."""
    pass


class NeuroMemoryServiceError(NeuroMemoryError):
    """Service error from the remote service."""
    
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ============================================================
# NeuroMemoryClient
# ============================================================


class NeuroMemoryClient:
    """Extended SDK client for MLSDM Neuro Memory Service.
    
    Provides a unified interface for:
    - Memory operations (append, query)
    - Decision-making with governance
    - Agent step protocol
    
    Supports both local (direct engine) and remote (HTTP API) modes.
    
    Args:
        mode: Operation mode - "local" for direct engine access, "remote" for HTTP API.
        base_url: Base URL for remote mode (e.g., "http://localhost:8000").
        backend: LLM backend for local mode ("local_stub" or "openai").
        api_key: API key for OpenAI backend (local mode) or auth (remote mode).
        timeout: Request timeout in seconds for remote mode.
        user_id: Default user ID for memory scoping.
        session_id: Default session ID for memory scoping.
        agent_id: Default agent ID for agent operations.
    
    Examples:
        >>> # Local mode with stub backend
        >>> client = NeuroMemoryClient(mode="local")
        >>> 
        >>> # Local mode with OpenAI
        >>> client = NeuroMemoryClient(mode="local", backend="openai", api_key="sk-...")
        >>> 
        >>> # Remote mode
        >>> client = NeuroMemoryClient(mode="remote", base_url="http://localhost:8000")
        >>> 
        >>> # Memory operations
        >>> result = client.append_memory("The user likes coffee")
        >>> items = client.query_memory("What does the user like?")
        >>> 
        >>> # Decision-making
        >>> decision = client.decide("Should I recommend coffee?", risk_level="low")
        >>> 
        >>> # Agent step
        >>> step = client.agent_step(
        ...     agent_id="assistant-1",
        ...     observation="User asked about coffee recommendations",
        ... )
    """
    
    def __init__(
        self,
        mode: Literal["local", "remote"] = "local",
        base_url: str | None = None,
        backend: Literal["local_stub", "openai"] = "local_stub",
        api_key: str | None = None,
        timeout: float = 30.0,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Initialize the NeuroMemoryClient."""
        self._mode = mode
        self._base_url = base_url or "http://localhost:8000"
        self._backend = backend
        self._api_key = api_key
        self._timeout = timeout
        self._user_id = user_id
        self._session_id = session_id
        self._agent_id = agent_id
        
        # Local mode: initialize engine
        self._engine: Any = None
        self._wrapper: Any = None
        self._embedding_fn: Any = None
        
        if mode == "local":
            self._init_local_engine()
        
        # Remote mode: initialize HTTP client
        self._http: requests.Session | None = None
        if mode == "remote":
            self._init_http_client()
    
    def _init_local_engine(self) -> None:
        """Initialize local engine for direct access."""
        from mlsdm import create_neuro_engine
        from mlsdm.engine import NeuroEngineConfig
        from mlsdm.engine.factory import build_stub_embedding_fn
        
        import os
        os.environ["LLM_BACKEND"] = self._backend
        if self._api_key and self._backend == "openai":
            os.environ["OPENAI_API_KEY"] = self._api_key
        
        config = NeuroEngineConfig(enable_fslgs=False, enable_metrics=True)
        self._engine = create_neuro_engine(config=config)
        self._wrapper = getattr(self._engine, "_mlsdm", None)
        self._embedding_fn = build_stub_embedding_fn(dim=config.dim)
    
    def _init_http_client(self) -> None:
        """Initialize HTTP client for remote access."""
        import requests
        self._http = requests.Session()
        self._http.headers.update({
            "Content-Type": "application/json",
        })
        if self._api_key:
            self._http.headers.update({
                "Authorization": f"Bearer {self._api_key}",
            })
    
    # ================================================================
    # Memory Operations
    # ================================================================
    
    def append_memory(
        self,
        content: str,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        moral_value: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryAppendResult:
        """Append content to cognitive memory.
        
        Args:
            content: Text content to store in memory.
            user_id: User identifier for scoping (overrides default).
            session_id: Session identifier for scoping (overrides default).
            agent_id: Agent identifier for scoping (overrides default).
            moral_value: Moral value for this memory entry (0.0-1.0).
            metadata: Additional metadata to store.
        
        Returns:
            MemoryAppendResult with operation status and memory ID.
        
        Raises:
            NeuroMemoryError: If operation fails.
        """
        user_id = user_id or self._user_id
        session_id = session_id or self._session_id
        agent_id = agent_id or self._agent_id
        
        if self._mode == "local":
            return self._append_memory_local(content, user_id, session_id, agent_id, moral_value, metadata)
        else:
            return self._append_memory_remote(content, user_id, session_id, agent_id, moral_value, metadata)
    
    def _append_memory_local(
        self,
        content: str,
        user_id: str | None,
        session_id: str | None,
        agent_id: str | None,
        moral_value: float,
        metadata: dict[str, Any] | None,
    ) -> MemoryAppendResult:
        """Append memory using local engine."""
        if self._wrapper is None:
            raise NeuroMemoryError("Local engine not initialized")
        
        timestamp = time.time()
        memory_id = hashlib.sha256(f"{content}{timestamp}".encode()).hexdigest()[:16]
        
        # Use wrapper to process content through cognitive pipeline
        result = self._wrapper.generate(
            prompt=content,
            moral_value=moral_value,
            max_tokens=1,
            context_top_k=0,
        )
        
        accepted = result.get("accepted", False)
        phase = result.get("phase", "unknown")
        
        state = self._wrapper.get_state()
        memory_stats = {
            "capacity": state.get("qilm_stats", {}).get("capacity", 0),
            "used": state.get("qilm_stats", {}).get("used", 0),
        }
        
        return MemoryAppendResult(
            success=accepted,
            memory_id=memory_id if accepted else None,
            phase=phase,
            accepted=accepted,
            memory_stats=memory_stats,
            message="Memory stored" if accepted else result.get("note", "Rejected"),
        )
    
    def _append_memory_remote(
        self,
        content: str,
        user_id: str | None,
        session_id: str | None,
        agent_id: str | None,
        moral_value: float,
        metadata: dict[str, Any] | None,
    ) -> MemoryAppendResult:
        """Append memory using remote HTTP API."""
        if self._http is None:
            raise NeuroMemoryError("HTTP client not initialized")
        
        import requests
        
        try:
            response = self._http.post(
                f"{self._base_url}/v1/memory/append",
                json={
                    "content": content,
                    "user_id": user_id,
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "moral_value": moral_value,
                    "metadata": metadata,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            return MemoryAppendResult(
                success=data.get("success", False),
                memory_id=data.get("memory_id"),
                phase=data.get("phase", "unknown"),
                accepted=data.get("accepted", False),
                memory_stats=data.get("memory_stats"),
                message=data.get("message"),
            )
        except requests.ConnectionError as e:
            raise NeuroMemoryConnectionError(f"Failed to connect: {e}") from e
        except requests.HTTPError as e:
            raise NeuroMemoryServiceError(
                f"Service error: {e}",
                status_code=e.response.status_code if e.response else None,
            ) from e
    
    def query_memory(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        top_k: int = 5,
        include_metadata: bool = True,
    ) -> MemoryQueryResult:
        """Query cognitive memory for relevant content.
        
        Args:
            query: Query text to search for.
            user_id: User identifier to scope query (overrides default).
            session_id: Session identifier to scope query (overrides default).
            agent_id: Agent identifier to scope query (overrides default).
            top_k: Number of results to retrieve.
            include_metadata: Whether to include metadata in results.
        
        Returns:
            MemoryQueryResult with matching memory items.
        
        Raises:
            NeuroMemoryError: If query fails.
        """
        user_id = user_id or self._user_id
        session_id = session_id or self._session_id
        agent_id = agent_id or self._agent_id
        
        if self._mode == "local":
            return self._query_memory_local(query, user_id, session_id, agent_id, top_k, include_metadata)
        else:
            return self._query_memory_remote(query, user_id, session_id, agent_id, top_k, include_metadata)
    
    def _query_memory_local(
        self,
        query: str,
        user_id: str | None,
        session_id: str | None,
        agent_id: str | None,
        top_k: int,
        include_metadata: bool,
    ) -> MemoryQueryResult:
        """Query memory using local engine."""
        if self._wrapper is None or self._embedding_fn is None:
            raise NeuroMemoryError("Local engine not initialized")
        
        # Access PELM directly from wrapper
        pelm = getattr(self._wrapper, "pelm", None)
        rhythm = getattr(self._wrapper, "rhythm", None)
        
        if pelm is None:
            raise NeuroMemoryError("PELM memory not available")
        
        query_vector = self._embedding_fn(query)
        
        # Get current phase for retrieval
        current_phase = 0.1 if rhythm and rhythm.is_wake() else 0.9
        phase_name = "wake" if current_phase == 0.1 else "sleep"
        
        # Retrieve from PELM
        try:
            retrievals = pelm.retrieve(
                query_vector.tolist(),
                current_phase=current_phase,
                phase_tolerance=0.15,
                top_k=top_k
            )
        except Exception as e:
            logger.warning(f"PELM retrieval failed: {e}")
            retrievals = []
        
        results = []
        for retrieval in retrievals:
            item = MemoryItem(
                content=str(retrieval.vector.tolist()[:5]) + "...",
                similarity=retrieval.resonance,  # Use resonance (cosine similarity)
                phase=retrieval.phase,
                metadata=None if not include_metadata else {},
            )
            results.append(item)
        
        return MemoryQueryResult(
            success=True,
            results=results,
            query_phase=phase_name,
            total_results=len(results),
            message=f"Retrieved {len(results)} items",
        )
    
    def _query_memory_remote(
        self,
        query: str,
        user_id: str | None,
        session_id: str | None,
        agent_id: str | None,
        top_k: int,
        include_metadata: bool,
    ) -> MemoryQueryResult:
        """Query memory using remote HTTP API."""
        if self._http is None:
            raise NeuroMemoryError("HTTP client not initialized")
        
        import requests
        
        try:
            response = self._http.post(
                f"{self._base_url}/v1/memory/query",
                json={
                    "query": query,
                    "user_id": user_id,
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "top_k": top_k,
                    "include_metadata": include_metadata,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            results = [
                MemoryItem(
                    content=item.get("content", ""),
                    similarity=item.get("similarity", 0.0),
                    phase=item.get("phase", 0.0),
                    metadata=item.get("metadata"),
                )
                for item in data.get("results", [])
            ]
            
            return MemoryQueryResult(
                success=data.get("success", False),
                results=results,
                query_phase=data.get("query_phase", "unknown"),
                total_results=data.get("total_results", 0),
                message=data.get("message"),
            )
        except requests.ConnectionError as e:
            raise NeuroMemoryConnectionError(f"Failed to connect: {e}") from e
        except requests.HTTPError as e:
            raise NeuroMemoryServiceError(
                f"Service error: {e}",
                status_code=e.response.status_code if e.response else None,
            ) from e
    
    # ================================================================
    # Decision Operations
    # ================================================================
    
    def decide(
        self,
        prompt: str,
        context: str | None = None,
        risk_level: Literal["low", "medium", "high", "critical"] = "medium",
        priority: Literal["low", "normal", "high", "urgent"] = "normal",
        mode: Literal["standard", "cautious", "confident", "emergency"] = "standard",
        max_tokens: int = 512,
        use_memory: bool = True,
        context_top_k: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> DecideResult:
        """Make a governed decision.
        
        Args:
            prompt: Input prompt for decision-making.
            context: Additional context for the decision.
            risk_level: Risk level affecting moral thresholds.
            priority: Priority level for processing.
            mode: Decision mode (standard, cautious, confident, emergency).
            max_tokens: Maximum tokens for response.
            use_memory: Whether to use memory for context.
            context_top_k: Number of memory items to retrieve.
            metadata: Additional metadata.
        
        Returns:
            DecideResult with decision and governance details.
        
        Raises:
            NeuroMemoryError: If decision fails.
        """
        if self._mode == "local":
            return self._decide_local(prompt, context, risk_level, priority, mode, max_tokens, use_memory, context_top_k, metadata)
        else:
            return self._decide_remote(prompt, context, risk_level, priority, mode, max_tokens, use_memory, context_top_k, metadata)
    
    def _get_moral_value_for_mode(self, mode: str, risk_level: str) -> float:
        """Calculate moral value based on mode and risk level."""
        mode_values = {"standard": 0.5, "cautious": 0.7, "confident": 0.4, "emergency": 0.3}
        risk_adjustments = {"low": -0.1, "medium": 0.0, "high": 0.1, "critical": 0.2}
        base = mode_values.get(mode, 0.5)
        adjustment = risk_adjustments.get(risk_level, 0.0)
        return min(1.0, max(0.0, base + adjustment))
    
    def _decide_local(
        self,
        prompt: str,
        context: str | None,
        risk_level: str,
        priority: str,
        mode: str,
        max_tokens: int,
        use_memory: bool,
        context_top_k: int,
        metadata: dict[str, Any] | None,
    ) -> DecideResult:
        """Make decision using local engine."""
        if self._engine is None:
            raise NeuroMemoryError("Local engine not initialized")
        
        timestamp = time.time()
        decision_id = hashlib.sha256(f"{prompt}{timestamp}".encode()).hexdigest()[:16]
        moral_value = self._get_moral_value_for_mode(mode, risk_level)
        
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
        
        result = self._engine.generate(
            prompt=full_prompt,
            max_tokens=max_tokens,
            moral_value=moral_value,
            context_top_k=context_top_k if use_memory else 0,
            user_intent="decision",
        )
        
        mlsdm_state = result.get("mlsdm", {})
        phase = mlsdm_state.get("phase", "unknown")
        rejected_at = result.get("rejected_at")
        accepted = rejected_at is None and result.get("error") is None and bool(result.get("response"))
        
        contour_decisions = [
            ContourDecision(
                contour="moral_filter",
                passed=accepted,
                score=moral_value,
                threshold=mlsdm_state.get("moral_threshold", 0.5),
                notes=f"mode={mode}, risk={risk_level}"
            ),
            ContourDecision(
                contour="risk_assessment",
                passed=risk_level != "critical" or mode == "emergency",
                score=0.8,
                threshold=0.5,
                notes=f"risk_level={risk_level}"
            ),
        ]
        
        return DecideResult(
            response=result.get("response", ""),
            accepted=accepted,
            phase=phase,
            contour_decisions=contour_decisions,
            memory_context_used=mlsdm_state.get("context_items", 0),
            risk_assessment={"level": risk_level, "mode": mode, "moral_value": moral_value},
            timing=result.get("timing"),
            decision_id=decision_id,
            message="Decision processed" if accepted else "Decision rejected",
        )
    
    def _decide_remote(
        self,
        prompt: str,
        context: str | None,
        risk_level: str,
        priority: str,
        mode: str,
        max_tokens: int,
        use_memory: bool,
        context_top_k: int,
        metadata: dict[str, Any] | None,
    ) -> DecideResult:
        """Make decision using remote HTTP API."""
        if self._http is None:
            raise NeuroMemoryError("HTTP client not initialized")
        
        import requests
        
        try:
            response = self._http.post(
                f"{self._base_url}/v1/decide",
                json={
                    "prompt": prompt,
                    "context": context,
                    "user_id": self._user_id,
                    "session_id": self._session_id,
                    "agent_id": self._agent_id,
                    "risk_level": risk_level,
                    "priority": priority,
                    "mode": mode,
                    "max_tokens": max_tokens,
                    "use_memory": use_memory,
                    "context_top_k": context_top_k,
                    "metadata": metadata,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            contour_decisions = [
                ContourDecision(
                    contour=cd.get("contour", ""),
                    passed=cd.get("passed", False),
                    score=cd.get("score"),
                    threshold=cd.get("threshold"),
                    notes=cd.get("notes"),
                )
                for cd in data.get("contour_decisions", [])
            ]
            
            return DecideResult(
                response=data.get("response", ""),
                accepted=data.get("accepted", False),
                phase=data.get("phase", "unknown"),
                contour_decisions=contour_decisions,
                memory_context_used=data.get("memory_context_used", 0),
                risk_assessment=data.get("risk_assessment"),
                timing=data.get("timing"),
                decision_id=data.get("decision_id"),
                message=data.get("message"),
            )
        except requests.ConnectionError as e:
            raise NeuroMemoryConnectionError(f"Failed to connect: {e}") from e
        except requests.HTTPError as e:
            raise NeuroMemoryServiceError(
                f"Service error: {e}",
                status_code=e.response.status_code if e.response else None,
            ) from e
    
    # ================================================================
    # Agent Operations
    # ================================================================
    
    def agent_step(
        self,
        agent_id: str,
        observation: str,
        user_id: str | None = None,
        session_id: str | None = None,
        internal_state: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        moral_value: float = 0.8,
    ) -> AgentStepResult:
        """Process a single step for an external agent.
        
        Args:
            agent_id: Unique agent identifier.
            observation: Current observation/input.
            user_id: User identifier (overrides default).
            session_id: Session identifier (overrides default).
            internal_state: Agent's internal state.
            tool_calls: Tool calls from previous step.
            tool_results: Results from tool calls.
            max_tokens: Maximum tokens for response.
            moral_value: Moral value for this step.
        
        Returns:
            AgentStepResult with action and updated state.
        
        Raises:
            NeuroMemoryError: If step fails.
        """
        user_id = user_id or self._user_id
        session_id = session_id or self._session_id
        
        if self._mode == "local":
            return self._agent_step_local(agent_id, observation, user_id, session_id, internal_state, tool_calls, tool_results, max_tokens, moral_value)
        else:
            return self._agent_step_remote(agent_id, observation, user_id, session_id, internal_state, tool_calls, tool_results, max_tokens, moral_value)
    
    def _agent_step_local(
        self,
        agent_id: str,
        observation: str,
        user_id: str | None,
        session_id: str | None,
        internal_state: dict[str, Any] | None,
        tool_calls: list[dict[str, Any]] | None,
        tool_results: list[dict[str, Any]] | None,
        max_tokens: int,
        moral_value: float,
    ) -> AgentStepResult:
        """Process agent step using local engine."""
        if self._engine is None:
            raise NeuroMemoryError("Local engine not initialized")
        
        timestamp = time.time()
        step_id = hashlib.sha256(f"{agent_id}:{observation}{timestamp}".encode()).hexdigest()[:16]
        
        prompt_parts = [f"Agent ID: {agent_id}"]
        if tool_results:
            prompt_parts.append("\nTool Results:")
            for result in tool_results:
                prompt_parts.append(f"- {result.get('tool', 'unknown')}: {result.get('result', '')}")
        prompt_parts.append(f"\nObservation: {observation}")
        prompt_parts.append("\nRespond with the next action.")
        
        result = self._engine.generate(
            prompt="\n".join(prompt_parts),
            max_tokens=max_tokens,
            moral_value=moral_value,
            context_top_k=5,
            user_intent="agent",
        )
        
        mlsdm_state = result.get("mlsdm", {})
        phase = mlsdm_state.get("phase", "unknown")
        rejected_at = result.get("rejected_at")
        accepted = rejected_at is None and result.get("error") is None and bool(result.get("response"))
        response_text = result.get("response", "")
        
        action_type: Literal["respond", "tool_call", "wait", "terminate"] = "respond"
        if not accepted:
            action_type = "wait"
        elif "terminate" in response_text.lower():
            action_type = "terminate"
        
        action = AgentAction(
            action_type=action_type,
            content=response_text if action_type == "respond" else None,
            tool_calls=None,
        )
        
        updated_state = internal_state or {}
        updated_state["last_step_id"] = step_id
        updated_state["step_count"] = updated_state.get("step_count", 0) + 1
        
        return AgentStepResult(
            action=action,
            response=response_text,
            phase=phase,
            accepted=accepted,
            updated_state=updated_state,
            memory_updated=accepted,
            memory_context_used=mlsdm_state.get("context_items", 0),
            step_id=step_id,
            timing=result.get("timing"),
            message="Step processed" if accepted else "Step rejected",
        )
    
    def _agent_step_remote(
        self,
        agent_id: str,
        observation: str,
        user_id: str | None,
        session_id: str | None,
        internal_state: dict[str, Any] | None,
        tool_calls: list[dict[str, Any]] | None,
        tool_results: list[dict[str, Any]] | None,
        max_tokens: int,
        moral_value: float,
    ) -> AgentStepResult:
        """Process agent step using remote HTTP API."""
        if self._http is None:
            raise NeuroMemoryError("HTTP client not initialized")
        
        import requests
        
        try:
            response = self._http.post(
                f"{self._base_url}/v1/agent/step",
                json={
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "observation": observation,
                    "internal_state": internal_state,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                    "max_tokens": max_tokens,
                    "moral_value": moral_value,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            action_data = data.get("action", {})
            action = AgentAction(
                action_type=action_data.get("action_type", "respond"),
                content=action_data.get("content"),
                tool_calls=action_data.get("tool_calls"),
            )
            
            return AgentStepResult(
                action=action,
                response=data.get("response", ""),
                phase=data.get("phase", "unknown"),
                accepted=data.get("accepted", False),
                updated_state=data.get("updated_state"),
                memory_updated=data.get("memory_updated", False),
                memory_context_used=data.get("memory_context_used", 0),
                step_id=data.get("step_id"),
                timing=data.get("timing"),
                message=data.get("message"),
            )
        except requests.ConnectionError as e:
            raise NeuroMemoryConnectionError(f"Failed to connect: {e}") from e
        except requests.HTTPError as e:
            raise NeuroMemoryServiceError(
                f"Service error: {e}",
                status_code=e.response.status_code if e.response else None,
            ) from e
    
    # ================================================================
    # Generation (for compatibility with NeuroCognitiveClient)
    # ================================================================
    
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        moral_value: float | None = None,
        context_top_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate a response (compatible with NeuroCognitiveClient).
        
        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            moral_value: Moral threshold value.
            context_top_k: Number of context items.
        
        Returns:
            Dictionary with response and metadata.
        """
        if self._mode == "local" and self._engine is not None:
            kwargs: dict[str, Any] = {"prompt": prompt}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if moral_value is not None:
                kwargs["moral_value"] = moral_value
            if context_top_k is not None:
                kwargs["context_top_k"] = context_top_k
            return self._engine.generate(**kwargs)
        else:
            # Remote mode: use /generate endpoint
            if self._http is None:
                raise NeuroMemoryError("HTTP client not initialized")
            
            import requests
            
            try:
                response = self._http.post(
                    f"{self._base_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "moral_value": moral_value,
                    },
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                raise NeuroMemoryError(f"Generate request failed: {e}") from e
    
    # ================================================================
    # Health Check
    # ================================================================
    
    def health_check(self) -> dict[str, Any]:
        """Check service health.
        
        Returns:
            Dictionary with health status.
        """
        if self._mode == "local":
            return {"status": "healthy", "mode": "local"}
        else:
            if self._http is None:
                raise NeuroMemoryError("HTTP client not initialized")
            
            import requests
            
            try:
                response = self._http.get(
                    f"{self._base_url}/health",
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()
                data["mode"] = "remote"
                return data
            except requests.RequestException as e:
                raise NeuroMemoryConnectionError(f"Health check failed: {e}") from e
    
    # ================================================================
    # Properties
    # ================================================================
    
    @property
    def mode(self) -> str:
        """Get current operation mode."""
        return self._mode
    
    @property
    def base_url(self) -> str:
        """Get base URL for remote mode."""
        return self._base_url
