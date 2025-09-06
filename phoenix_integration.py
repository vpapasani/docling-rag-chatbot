import random  # Added for generating dummy data
import time
import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# 🔹 Added for Playground tracing

from openinference.instrumentation.openai import OpenAIInstrumentor

# 1. Define the Phoenix OTLP HTTP endpoint directly
# IMPORTANT: This should be the actual OTLP collector endpoint, NOT the UI endpoint.
PHOENIX_COLLECTOR_URL = "http://localhost:4318/v1/traces"  # Set the actual URL here

# This is what Phoenix uses to group your traces into a project.
SERVICE_NAME = "rag-chat-project"  # Set your desired project name here

# Resource for Phoenix project
# Use the directly defined SERVICE_NAME
resource = Resource.create({"service.name": SERVICE_NAME})

provider = TracerProvider(resource=resource)

# Use SimpleSpanProcessor to flush immediately per span
# For high-throughput scenarios, consider BatchSpanProcessor for performance
# Use the directly defined PHOENIX_COLLECTOR_URL
exporter = OTLPSpanExporter(endpoint=PHOENIX_COLLECTOR_URL)
processor = SimpleSpanProcessor(exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
# Use the SERVICE_NAME for the tracer name, or another descriptive name
tracer = trace.get_tracer(SERVICE_NAME)
# 🔹 Enable OpenAI SDK auto-instrumentation (Playground uses OpenAI SDK)
OpenAIInstrumentor().instrument()

def trace_rag_session(session_id: str, query: str, top_ctx: list[str], answer: str, elapsed: float,model_name:str):
    """
    Trace RAG session synchronously to Phoenix.
    """

    # Root span
    with tracer.start_as_current_span(f"RAG Session {session_id}") as span:
        span.set_attribute("rag.session_id", session_id)
        span.set_attribute("rag.query", query)
        span.set_attribute("rag.answer", answer)
        span.set_attribute("rag.elapsed_ms", elapsed)
        span.set_attribute("rag.model_used",model_name)

        # Child spans for context
        for idx, ctx_text in enumerate(top_ctx):
            with tracer.start_as_current_span(f"Context Chunk {idx + 1}") as ctx_span:
                ctx_span.set_attribute("rag.context.index", idx + 1)
                ctx_span.set_attribute("rag.context.text", ctx_text)

    # Force flush synchronously so Phoenix gets the data immediately
    provider.force_flush()


if __name__ == "__main__":
    print("Starting RAG session tracing test...")
    trace_rag_session(
        session_id=f"session_{random.randint(10000, 99999)}",
        query="What is the capital of France?",
        top_ctx=["Paris is the capital of France.", "France is in Western Europe."],
        answer="The capital of France is Paris.",
        elapsed=random.uniform(50.0, 500.0)
    )
    print("RAG session traced successfully (if collector is running). Check your Phoenix UI.")
