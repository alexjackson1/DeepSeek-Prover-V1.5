from contextlib import asynccontextmanager
from typing import Any, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from prover.lean.verifier import Lean4ServerScheduler
from prover.utils import LEAN4_DEFAULT_HEADER as DEFAULT_HEADER


class ProveRequest(BaseModel):
    name: Optional[str] = None
    model: str = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
    prompt: str = r"Complete the following Lean 4 code:\n\n```lean4"
    informal_prefix: Optional[str] = None
    header: str = DEFAULT_HEADER
    goal: Optional[str] = None
    formal_statement: str


class ParserMessage(BaseModel):
    severity: str
    pos: dict
    endPos: dict
    data: str


class LeanOutput(BaseModel):
    complete: bool
    sorries: List[str] = []
    tactics: List[str] = []
    errors: List[ParserMessage] = []
    warnings: List[ParserMessage] = []
    infos: List[ParserMessage] = []
    system_messages: str = ""
    system_errors: Optional[List[Any]] = None
    ast: Any = {}
    verified_code: str
    verify_time: float


class ProveResponse(BaseModel):
    success: bool
    message: str
    full_prompt: str
    completion: str
    data: Optional[LeanOutput] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # check cuda
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    app.state.tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-Prover-V1.5-RL"
    )
    app.state.model = LLM(
        model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
        max_num_batched_tokens=8192,
        seed=1,
        trust_remote_code=True,
    )
    app.state.lean4_scheduler = Lean4ServerScheduler(
        name="verifier",
        max_concurrent_requests=1,
        timeout=300,
        memory_limit=10,
    )
    yield
    if hasattr(app.state, "lean4_scheduler"):
        app.state.lean4_scheduler.close()


app = FastAPI(
    title="dsprove-api",
    description="API for the dsprove-prover service (based on DeepSeek-Prover-V1.5)",
    version="0.1.0",
    lifespan=lifespan,
)


origins = [
    "http://localhost:5173",
    "http://dsprove.knowledgebase.systems",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST"],
    allow_headers=["*"],
)


def make_prompt(request: ProveRequest) -> str:
    comment = "\n".join([f"\\- {l}" for l in request.informal_prefix.splitlines()])
    return f"{request.prompt}\n{request.header}\n{comment}\n{request.formal_statement}"


def process_prove_request(request: ProveRequest) -> Tuple[str, str, LeanOutput]:
    complete_prompt = make_prompt(request)

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=2048,
        top_p=0.95,
        n=1,
    )

    model_outputs = app.state.model.generate(
        [complete_prompt], sampling_params, use_tqdm=False
    )
    output_text = model_outputs[0].outputs[0].text

    completion = complete_prompt + output_text

    lean_blocks = re.search(r"```lean4\n(.*?)\n```", completion, re.DOTALL).group(1)
    request_id_list = app.state.lean4_scheduler.submit_all_request([lean_blocks])

    outputs_list: List[Any] = app.state.lean4_scheduler.get_all_request_outputs(
        request_id_list
    )
    lean_output = outputs_list[0]

    return complete_prompt, completion, lean_output


@app.post("/prove", response_model=ProveResponse)
async def prove(request: ProveRequest):
    try:
        complete_prompt, completion, lean_output = process_prove_request(request)
        success = lean_output["pass"] and lean_output["complete"]
        message = "Proved" if success else "Failed to prove"
        return ProveResponse(
            success=success,
            message=message,
            full_prompt=complete_prompt,
            completion=completion,
            data=lean_output,
        )
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(
            status_code=503, detail="GPU memory exhausted - please try again later"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
