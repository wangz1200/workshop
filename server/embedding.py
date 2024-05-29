from typing import List, Any
import fastapi as fa
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

import shared


class ResModel(BaseModel):
    code: int = 0
    msg: str = ""
    data: Any = None


class ModelPermission(BaseModel):
    id: str = "0"
    object: str = "model_permission"
    created: int = 0
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelCard(BaseModel):

    id: str = "text2vec"
    object: str = "model"
    created: int = 0
    owned_by: str = "gxllm"
    root: str | None = None
    parent: str | None = None
    permission: List[ModelPermission] | None = None


class ModelCardList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class EmbeddingReq(BaseModel):
    text: str | List[str] | None = None
    input: str | List[str] | None = None
    model: str | None = None


model = shared.embedding.Embedding(
    model_name_or_path="../models/text2vec-base-chinese",
    device="cpu",
)


app = fa.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
def get_models():
    model_cards = [
        ModelCard(),
    ]
    model_list = ModelCardList(
        data=model_cards
    )
    return JSONResponse(content=model_list.model_dump())


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingReq):
    res = ResModel()
    try:
        text = req.text or req.input
        text = text if isinstance(text, list) else [text]
        res.data = model.encode(text)
    except Exception as ex:
        res.code = -1
        res.msg = str(ex)
    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
