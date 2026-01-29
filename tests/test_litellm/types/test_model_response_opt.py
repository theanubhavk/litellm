import pytest
import time
import sys
import json
from litellm.types.utils import Choices, Message, Usage, ModelResponse, ModelResponseStream

# def test_model_response_stream_chunk_access():
#     """
#     Verifies that streaming chunks are handled correctly.
#     """
#     stream_chunk = ModelResponseStream(
#         id="chatcmpl-stream-123",
#         choices=[
#             Choices(
#                 finish_reason=None,
#                 index=0,
#                 delta=Message(content="Hello", role="assistant")
#             )
#         ],
#         created=1677652288,
#         model="gpt-4"
#     )

#     # Attribute access
#     assert stream_chunk.choices[0].delta.content == "Hello"
    
#     # Dict access
#     assert stream_chunk["choices"][0]["delta"]["content"] == "Hello"

# ----------------------------------------------------------------------
# 1. Functional Correctness Tests
#   Ensure the optimizations didn't break standard usage.
# ----------------------------------------------------------------------

def test_model_response_initialization_and_access():
    """
    Verifies that ModelResponse allows both attribute-style (obj.x) 
    and dictionary-style (obj['x']) access.
    """
    # Simulate a standard response payload
    response = ModelResponse(
        id="chatcmpl-123",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="Hello world", role="assistant")
            )
        ],
        created=1677652288,
        model="gpt-3.5-turbo",
        object="chat.completion",
        usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )

    # Test Attribute Access
    assert response.id == "chatcmpl-123"
    assert response.choices[0].message.content == "Hello world"
    assert response.usage.total_tokens == 20

    # Test Dictionary Access (Backwards Compatibility)
    assert response["id"] == "chatcmpl-123"
    assert response["choices"][0]["message"]["content"] == "Hello world"
    
    # Test .get() method if supported
    if hasattr(response, "get"):
        assert response.get("model") == "gpt-3.5-turbo"


def test_json_serialization():
    """
    Ensures the object can still be serialized to JSON, which is critical for API responses.
    """
    response = ModelResponse(
        id="test-id",
        choices=[Choices(message=Message(content="test"), index=0, finish_reason="stop")],
        model="test-model"
    )

    # Method 1: Pydantic style .json() or .model_dump_json()
    if hasattr(response, "model_dump_json"):
        json_str = response.model_dump_json()
    elif hasattr(response, "json"):
        json_str = response.json()
    else:
        # Fallback to json.dumps on the dict representation
        json_str = json.dumps(response.dict() if hasattr(response, "dict") else dict(response))

    assert "test-id" in json_str
    assert "test-model" in json_str

# ----------------------------------------------------------------------
# 2. Performance Benchmarks
#   Prove your optimizations worked.
# ----------------------------------------------------------------------

def test_instantiation_performance(benchmark=None):
    """
    Benchmarks the creation of 100,000 separate ModelResponse objects.
    """
    iterations = 100_000
    
    # 1. PREPARE DATA (Outside the timer)
    # We want to test the ModelResponse logic, not how fast we can make 'Choices'
    # Use simple dicts if you want to test the parsing logic, 
    # or pre-instantiated objects if testing just the wrapper.
    # Since your optimization targets 'choices' validation, passing dicts is the most realistic test.
    single_choice_payload = [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {"content": "Hello", "role": "assistant"}
        }
    ]

    # --- WARM UP PHASE ---
    _ = ModelResponse(
        id="warmup",
        choices=[], # Test empty
        created=0,
        model="warmup"
    )
    # Warm up with data to trigger the specific path you optimized
    _ = ModelResponse(
        id="warmup",
        choices=single_choice_payload, 
        created=0,
        model="warmup"
    )
    # ---------------------

    # 2. BENCHMARK LOOP
    start_time = time.perf_counter()

    for i in range(iterations):
        # We create a NEW ModelResponse every time (simulating real requests)
        _ = ModelResponse(
            id="id-123",
            choices=single_choice_payload, # Pass the SAME list to save setup time
            created=1234567890,
            model="gpt-3.5-optimized"
        )
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations

    print(f"\n[Performance] Instantiated {iterations} ModelResponses in {total_time:.4f}s")
    print(f"[Performance] Average time per object: {avg_time * 1e6:.2f} microseconds")

    assert total_time < 5.0, "Instantiation is too slow!"
    

def test_memory_footprint():
    """
    Checks the size of the object in memory. 
    Useful if you added __slots__ to ModelResponse.
    """
    # Warm up here as well just in case
    _ = ModelResponse(id="warmup", choices=[], created=0, model="warmup")

    response = ModelResponse(
        id="mem-test",
        choices=[],
        created=1234567890,
        model="gpt-3.5-turbo"
    )

    size = sys.getsizeof(response)
    print(f"\n[Memory] Size of single ModelResponse object: {size} bytes")

    # If you implemented __slots__, this should not have a __dict__
    # Uncomment the following line if your optimization involved __slots__
    # assert not hasattr(response, "__dict__"), "Object still has __dict__, __slots__ not applied correctly"

# ----------------------------------------------------------------------
# 3. Streaming Integration Test
# ----------------------------------------------------------------------

def test_streaming_accumulation():
    """
    Simulates a full streaming flow to ensure the optimized class 
    handles rapid updates correctly.
    """
    chunks = ["Hello", " ", "World", "!"]
    full_content = ""

    for text in chunks:
        chunk = ModelResponseStream(
            choices=[Choices(delta=Message(content=text), index=0)]
        )
        # Simulate extraction logic common in litellm
        delta = chunk.choices[0].delta.content
        if delta:
            full_content += delta

    assert full_content == "Hello World!"