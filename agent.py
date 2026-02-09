import json
from typing import Optional

from ag_ui.core import RunAgentInput
from agent_loop import agent_loop
from agui import stream_agui_events, to_openai_messages, to_sse_data
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

# Try to import tools, fallback to empty list if not available
try:
    from tools import MAIN_AGENT_TOOLS
except ImportError:
    MAIN_AGENT_TOOLS = []

app = FastAPI()

class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {"question": "What is the weather in Beijing today?"}
        },
    )

    question: str
    chat_history: Optional[list] = None

    def to_messages(self) -> list:
        if self.chat_history:
            return self.chat_history + [{"role": "user", "content": self.question}]
        else:
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.question},
            ]


class QueryResponse(BaseModel):
    answer: str


@app.post("/")
async def query(req: QueryRequest) -> QueryResponse:
    """
    Basic LLM API example.

    Invoke example:

    ```
    curl -X POST "http://localhost:8000/" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the weather in Beijing today?"}'

    ```

    Response example:

    ```json
    {
        "answer": "Beijing has sunny weather today, with temperatures between 10°C and 20°C."
    }
    ```


    """

    result = ""

    # Return messages after the last tool call message as the final answer
    async for chunk in agent_loop(req.to_messages(), MAIN_AGENT_TOOLS):
        if chunk.type == "tool_call" or chunk.type == "tool_call_result":
            result = ""
        elif chunk.type == "text" and chunk.content:
            result += chunk.content

    return QueryResponse(answer=result)


@app.post("/stream")
async def stream(req: QueryRequest) -> StreamingResponse:
    """
    Streaming query example.
    Invoke example:

    ```shell
    curl -N -X POST "http://localhost:8000/stream" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the weather in Beijing today?"}'

    ```

    Response example:

    ```text

    data: {"answer": "Beijing has "}

    data: {"answer": "sunny weather"}

    data: {"answer": " today, with"}

    data: {"answer": " temperatures"}

    data: {"answer": " between 10°C and 20°C."}


    ```

    """

    async def stream_response():
        async for chunk in agent_loop(req.to_messages(), MAIN_AGENT_TOOLS):
            if chunk.type == "text" and chunk.content:
                data = {
                    "answer": chunk.content,
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )


@app.post("/ag-ui")
async def ag_ui(run_agent_input: RunAgentInput) -> StreamingResponse:
    """
    AG-UI Protocol endpoint for streaming LLM interactions.

    AG-UI Protocol: https://docs.ag-ui.com/introduction
    """

    messages = to_openai_messages(run_agent_input.messages)

    async def stream_response():
        async for event in stream_agui_events(
            chunks=agent_loop(messages, MAIN_AGENT_TOOLS), run_agent_input=run_agent_input
        ):
            yield to_sse_data(event)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )
