"""Offline check that the installed google-genai SDK can convert every schema we send to Gemini."""

import pytest
from google import genai
from google.genai import _transformers as tr

from app.schemas.evaluation import EvaluationLLM
from app.schemas.plagiarism import ParaphraseLLM
from app.schemas.questions import GeneratedQuestionsLLM
from app.schemas.rag import RagAnswerLLM


@pytest.mark.parametrize("schema", [GeneratedQuestionsLLM, EvaluationLLM, ParaphraseLLM, RagAnswerLLM])
def test_sdk_accepts_response_schema(schema):
    client = genai.Client(api_key="dummy")
    converted = tr.t_schema(client._api_client, schema)
    assert converted is not None and converted.type is not None
