"""
INTEGRATION TESTS – REAL PROVIDERS ONLY

This test file validates that:
- Anthropic, OpenAI, and Google Gemini clients can be instantiated with real APIs
- Single-shot generation works
- Batch processing works
- Batch requests can be loaded from disk
- Batch results are downloaded and persisted to JSON
- Example batch files are auto-created when missing

REQUIRES:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- GOOGLE_API_KEY

Run manually or with pytest:
    pytest tests/test_llm_real_providers.py -s
    
To cleanup stuck Gemini jobs:
    CLEANUP_GEMINI_JOBS=1 pytest tests/test_llm_real_providers.py::test_batch_submission -s
"""

import json
import os
import pathlib
import time
from typing import Dict, Optional

import pytest

from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient, BatchHandle



BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "batch_data"
RESULTS_DIR = BASE_DIR / "batch_results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TEST_PROMPT = "Explain the difference between a list and a tuple in Python in one sentence."

PROVIDERS = ["anthropic", "openai", "google"]

# Maximum number of active Gemini jobs before we refuse to submit (to prevent queue saturation)
MAX_GEMINI_QUEUE_DEPTH = 5


def example_batch_requests(provider: str) -> list[dict]:
    """
    Returns provider-specific example requests.
    """
    if provider == "google":
        return [
            {
                "id": "request_1",
                "prompt": "Explain how AI works in a few words"
            },
            {
                "id": "request_2", 
                "prompt": "Explain how quantum computing works in a few words"
            },
            {
                "id": "request_3",
                "prompt": "What is machine learning?"
            },
            {
                "id": "request_4",
                "prompt": "Explain neural networks simply"
            },
            {
                "id": "request_5",
                "prompt": "What is deep learning?"
            },
            {
                "id": "request_6",
                "prompt": "Explain natural language processing"
            }
        ]
    
    return [
        {
            "id": "req-1",
            "prompt": "What is Python?"
        },
        {
            "id": "req-2",
            "prompt": "What is an LLM?"
        }
    ]

def ensure_batch_file(provider: str) -> pathlib.Path:
    """
    Ensure that a JSON batch file exists for the given provider.
    If missing, create it using example_batch_requests().
    """
    path = batch_file_path(provider)
    if not path.exists():
        requests = example_batch_requests(provider)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(requests, f, indent=2)
        print(f"[INFO] Created example batch file for {provider} → {path}")
    return path

def batch_file_path(provider: str) -> pathlib.Path:
    return DATA_DIR / f"{provider}_batch_requests.json"


def batch_results_path(provider: str) -> pathlib.Path:
    return RESULTS_DIR / f"{provider}_batch_results.json"


def batch_id_path(provider: str) -> pathlib.Path:
    """Path to store batch ID for later retrieval"""
    return DATA_DIR / f"{provider}_batch_id.txt"

def load_batch_requests(provider: str) -> list[dict]:
    """
    Load batch requests from file, creating an example file if missing.
    """
    path = ensure_batch_file(provider)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_batch_results(provider: str, results: Dict[str, str]) -> None:
    path = batch_results_path(provider)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved {provider} batch results → {path}")


def save_batch_id(provider: str, batch_id: str) -> None:
    """Save batch ID for later retrieval"""
    path = batch_id_path(provider)
    with open(path, "w", encoding="utf-8") as f:
        f.write(batch_id)
    print(f"[INFO] Saved {provider} batch ID → {path}")


def load_batch_id(provider: str) -> str:
    """Load previously saved batch ID"""
    path = batch_id_path(provider)
    if not path.exists():
        raise FileNotFoundError(
            f"No batch ID found for {provider}. "
            f"Run test_batch_submission first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def cleanup_gemini_queue(client: UnifiedLLMClient, dry_run: bool = True, max_active: int = MAX_GEMINI_QUEUE_DEPTH) -> dict:
    """
    Check Gemini batch queue depth and optionally cancel stuck jobs.
    
    Set CLEANUP_GEMINI_JOBS=1 environment variable to actually cancel jobs.
    Otherwise just warns about queue depth.
    """
    if client.provider != "google":
        return {'skipped': True}
    
    print(f"\n[INFO] Checking Gemini batch queue status...")
    
    try:
        # List all jobs
        jobs = list(client.client.batches.list(config={'page_size': 100}))
    except Exception as e:
        print(f"[WARNING] Could not list batch jobs: {e}")
        return {'error': str(e)}
    
    active_states = {'QUEUED', 'RUNNING', 'PENDING', 'JOB_STATE_QUEUED', 'JOB_STATE_RUNNING', 'JOB_STATE_PENDING'}
    terminal_states = {'SUCCEEDED', 'FAILED', 'CANCELLED', 'EXPIRED', 'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'}
    
    active_jobs = []
    terminal_jobs = []
    
    for job in jobs:
        state = getattr(job.state, 'name', str(job.state))
        if state in active_states:
            active_jobs.append((job.name, state, job.create_time))
        elif state in terminal_states:
            terminal_jobs.append(job.name)
    
    result = {
        'total': len(jobs),
        'active': len(active_jobs),
        'terminal': len(terminal_jobs),
        'cancelled': 0,
        'dry_run': dry_run
    }
    
    print(f"[INFO] Queue status: {len(active_jobs)} active, {len(terminal_jobs)} terminal (of {len(jobs)} total)")
    
    # Show active jobs
    if active_jobs:
        print(f"[INFO] Active jobs:")
        for name, state, created in active_jobs[:10]:  # Show first 10
            print(f"  - {name} ({state}, created: {created})")
        if len(active_jobs) > 10:
            print(f"  ... and {len(active_jobs) - 10} more")
    
    # Check if we should cleanup
    if len(active_jobs) >= max_active:
        print(f"[WARNING] Queue saturated: {len(active_jobs)} active jobs (max recommended: {max_active})")
        
        if dry_run:
            print(f"[WARNING] Set CLEANUP_GEMINI_JOBS=1 to cancel active jobs, or wait for completion")
            print(f"[WARNING] New submissions may hang indefinitely until queue clears")
            return result
        
        # Actually cancel jobs (newest first to clear queue faster)
        print(f"[INFO] Cancelling {len(active_jobs)} active jobs...")
        for name, state, _ in reversed(active_jobs):
            try:
                client.client.batches.cancel(name=name)
                print(f"  ✓ Cancelled {name}")
                result['cancelled'] += 1
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"  ✗ Failed to cancel {name}: {e}")
    
    return result


@pytest.mark.parametrize("provider", PROVIDERS)
def test_single_shot_generation(provider: str):
    """
    Validate single-shot generation with real providers.
    """
    client = UnifiedLLMClient(provider=provider)

    result = client.generate(
        prompt=TEST_PROMPT,
        max_tokens=100,
    )

    assert isinstance(result, dict)
    assert "content" in result
    assert isinstance(result["content"], str)
    assert len(result["content"]) > 0

    print(f"\n[{provider.upper()} SINGLE-SHOT RESULT]\n{result['content']}\n")

@pytest.mark.integration
def test_integration_marker():
    assert True


@pytest.mark.integration
@pytest.mark.batch
@pytest.mark.parametrize("provider", PROVIDERS)
def test_batch_submission(provider: str):
    """
    Submit batch - uses parallel single-shot for small Gemini batches
    to avoid 15-minute queue delays.
    """
    client = UnifiedLLMClient(provider=provider)
    requests = load_batch_requests(provider)
    assert len(requests) > 0

    # GEMINI SMALL BATCH OPTIMIZATION
    if provider == "google" and len(requests) <= 5:
        print(f"\n⚡ Using PARALLEL single-shot for {len(requests)} prompts (bypass broken Batch API)")
        handle = client.submit_gemini_parallel(requests)
    else:
        # Standard file-based batch for larger jobs (or other providers)
        jsonl_path = str(DATA_DIR / f"{provider}_batch_input.jsonl")
        handle = client.submit_batch(requests=requests, jsonl_path=jsonl_path)

    assert handle.id
    assert handle.provider == provider
    save_batch_id(provider, handle.id)
    
    print(f"[INFO] Submitted {provider} batch: {handle.id}")
    
    # If parallel, save results immediately since they're ready
    if handle.metadata and handle.metadata.get("is_parallel"):
        print(f"[INFO] Results ready immediately (parallel execution)")
        save_batch_results(provider, handle.metadata["results"])
        print(f"[INFO] Saved results to: {batch_results_path(provider)}")


@pytest.mark.integration
@pytest.mark.batch
@pytest.mark.veryslow
@pytest.mark.parametrize("provider", PROVIDERS)
def test_batch_wait_and_retrieve(provider: str):
    """Retrieve results - handles both parallel (instant) and regular (slow) batches."""
    client = UnifiedLLMClient(provider=provider)
    
    try:
        batch_id = load_batch_id(provider)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    
    print(f"\n{'='*60}")
    print(f"RETRIEVING BATCH: {batch_id}")
    print(f"{'='*60}")
    
    handle = BatchHandle(provider=provider, id=batch_id)
    
    # Check if this is a parallel batch (results already computed)
    # FIX: Check both metadata AND batch_id pattern since metadata is lost on reload
    is_parallel = handle.metadata and handle.metadata.get("is_parallel")
    is_parallel_id = batch_id.startswith("parallel-")
    
    if is_parallel or is_parallel_id:
        print(f"\n⚡ Fast retrieval (parallel batch - results pre-computed)")
        
        # For parallel batches, results were already saved during submission
        # Just load them from file
        results_path = batch_results_path(provider)
        if results_path.exists():
            print(f"[INFO] Loading pre-saved results from: {results_path}")
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            # Fallback: try to call the method if metadata somehow preserved
            if hasattr(client, 'retrieve_gemini_parallel_results'):
                results = client.retrieve_gemini_parallel_results(handle)
            else:
                pytest.fail(f"Parallel batch results not found in {results_path}")
    else:
        # Regular Batch API with extended timeout
        print(f"\n⏳ Slow retrieval (Batch API - may take 15+ min)...")
        timeout = 600 if provider == "google" else 1800
        try:
            results = client.retrieve_batch_results(handle, timeout=timeout)
        except TimeoutError as e:
            print(f"\n❌ Batch API timed out after {timeout}s")
            pytest.fail(f"Batch API unreliable: {e}")
    
    # Validation
    assert isinstance(results, dict)
    assert len(results) > 0
    
    for req_id, content in results.items():
        assert isinstance(content, str)
        assert content.strip(), f"Empty output for {req_id}"
        assert "[ERROR]" not in content, f"Error in {req_id}: {content}"
    
    save_batch_results(provider, results)
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS: Retrieved {len(results)} results")
    print(f"{'='*60}")


@pytest.mark.integration
def test_cleanup_stuck_gemini_batch():
    """
    Cancel the stuck batch from previous failed attempts.
    Run this to clean up before switching to parallel mode.
    """
    provider = "google"
    client = UnifiedLLMClient(provider=provider)
    
    try:
        batch_id = load_batch_id(provider)
        print(f"Cancelling stuck batch: {batch_id}")
        client.client.batches.cancel(name=batch_id)
        print(f"✅ Cancelled successfully")
        
        # Remove the ID file so we can submit fresh
        batch_id_path(provider).unlink()
        print(f"✅ Removed stale batch ID file")
    except FileNotFoundError:
        print(f"No batch ID file found - nothing to cancel")
    except Exception as e:
        print(f"Could not cancel (may already be terminal): {e}")

@pytest.mark.integration
def test_check_gemini_queue():
    """
    Check Gemini batch queue status.
    Run standalone to see how many jobs are queued/running.
    
    Usage:
        pytest test_llm_client.py::test_check_gemini_queue -s
    """
    provider = "google"
    client = UnifiedLLMClient(provider=provider)
    
    print(f"\n{'='*60}")
    print(f"GEMINI BATCH QUEUE STATUS")
    print(f"{'='*60}")
    
    try:
        print("\n📡 Fetching jobs from API...")
        jobs = list(client.client.batches.list(config={'page_size': 100}))
    except Exception as e:
        pytest.fail(f"Failed to fetch jobs: {e}")
    
    # Categorize by state
    active_states = {'QUEUED', 'RUNNING', 'PENDING', 'JOB_STATE_QUEUED', 'JOB_STATE_RUNNING', 'JOB_STATE_PENDING'}
    terminal_states = {'SUCCEEDED', 'FAILED', 'CANCELLED', 'EXPIRED', 'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'}
    
    queued = []
    running = []
    succeeded = []
    failed = []
    other = []
    
    for job in jobs:
        state = getattr(job.state, 'name', str(job.state))
        
        if state in ('QUEUED', 'JOB_STATE_QUEUED'):
            queued.append(job)
        elif state in ('RUNNING', 'JOB_STATE_RUNNING'):
            running.append(job)
        elif state in ('SUCCEEDED', 'JOB_STATE_SUCCEEDED'):
            succeeded.append(job)
        elif state in ('FAILED', 'CANCELLED', 'EXPIRED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
            failed.append(job)
        else:
            other.append((job, state))
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"  🕐 QUEUED (waiting):   {len(queued)}")
    print(f"  ⚙️  RUNNING:           {len(running)}")
    print(f"  ✅ SUCCEEDED:          {len(succeeded)}")
    print(f"  ❌ FAILED/CANCELLED:   {len(failed)}")
    print(f"  ──────────────────────────")
    print(f"  📈 TOTAL:              {len(jobs)}")
    
    active_count = len(queued) + len(running)
    print(f"\n🚨 ACTIVE (blocking new jobs): {active_count}")
    
    if active_count >= 5:
        print(f"   ⚠️  WARNING: Queue saturated! New submissions will hang.")
        print(f"   💡 Run with CLEANUP_GEMINI_JOBS=1 to cancel active jobs.")
    
    # Show queued jobs (the ones blocking you)
    if queued:
        print(f"\n🕐 QUEUED JOBS (these are blocking your new submissions):")
        for i, job in enumerate(reversed(queued[-10:]), 1):  # Show last 10
            created = getattr(job, 'create_time', 'unknown')
            print(f"   {i}. {job.name}")
            print(f"      Created: {created}")
    
    if running:
        print(f"\n⚙️  RUNNING JOBS:")
        for job in running:
            print(f"   • {job.name}")
    
    if failed and len(failed) > 0:
        print(f"\n🗑️  RECENT FAILED/CANCELLED (last 3):")
        for job in failed[-3:]:
            state = getattr(job.state, 'name', str(job.state))
            print(f"   • {job.name} ({state})")
    
    print(f"\n{'='*60}")
    print(f"Check complete. Queue depth: {active_count} active jobs")
    print(f"{'='*60}\n")
    
    # Return counts for assertions if needed
    assert True

@pytest.mark.integration
def test_check_current_gemini_batch():
    """
    Check the status of the CURRENT batch (from saved file) specifically.
    This helps debug why retrieval hangs.
    
    Usage:
        pytest test_llm_client.py::test_check_current_gemini_batch -s
    """
    provider = "google"
    client = UnifiedLLMClient(provider=provider)
    
    print(f"\n{'='*60}")
    print(f"CHECKING CURRENT BATCH STATUS")
    print(f"{'='*60}")
    
    # Load the current batch ID
    try:
        batch_id = load_batch_id(provider)
        print(f"📄 Saved batch ID: {batch_id}")
    except FileNotFoundError:
        print(f"❌ No batch ID file found. Run test_batch_submission first.")
        pytest.skip("No batch submitted yet")
        return
    
    # Try to fetch this specific batch
    print(f"\n🔍 Fetching specific batch from API...")
    try:
        # Import here to avoid issues if not google
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
        
        def get_batch():
            return client.client.batches.get(name=batch_id)
        
        # Timeout after 10s to avoid hanging
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_batch)
            batch_job = future.result(timeout=10)
            
        state = getattr(batch_job.state, 'name', str(batch_job.state))
        print(f"✅ Found batch!")
        print(f"   State: {state}")
        print(f"   Name: {batch_job.name}")
        print(f"   Create time: {batch_job.create_time}")
        
        # Check specific terminal conditions
        if state in ('SUCCEEDED', 'JOB_STATE_SUCCEEDED'):
            print(f"\n🎉 Batch COMPLETED successfully!")
            print(f"   Result file: {getattr(batch_job.dest, 'file_name', 'N/A')}")
        elif state in ('FAILED', 'JOB_STATE_FAILED'):
            error = getattr(batch_job, 'error', 'Unknown error')
            print(f"\n❌ Batch FAILED: {error}")
        elif state in ('CANCELLED', 'JOB_STATE_CANCELLED'):
            print(f"\n🚫 Batch was CANCELLED")
        elif state in ('PENDING', 'QUEUED', 'JOB_STATE_PENDING', 'JOB_STATE_QUEUED'):
            print(f"\n⏳ Batch is WAITING in queue (this is why retrieval hangs)")
            print(f"   It may take 10-15 minutes to start processing.")
        elif state in ('RUNNING', 'JOB_STATE_RUNNING'):
            print(f"\n⚙️  Batch is currently RUNNING")
        else:
            print(f"\n⚠️  Unknown state: {state}")
            
    except FutureTimeout:
        print(f"❌ Timeout fetching batch after 10s")
        print(f"   The batch ID may be invalid or the API is unresponsive.")
    except Exception as e:
        print(f"❌ Error fetching batch: {e}")
        print(f"   This usually means the batch ID is invalid or expired.")
    
    # Also show overall queue context
    print(f"\n{'='*60}")
    print(f"QUEUE CONTEXT (for comparison)")
    print(f"{'='*60}")
    
    all_jobs = list(client.client.batches.list(config={'page_size': 100}))
    active = [j for j in all_jobs if getattr(j.state, 'name', str(j.state)) 
              in ('QUEUED', 'RUNNING', 'PENDING', 'JOB_STATE_QUEUED', 'JOB_STATE_RUNNING', 'JOB_STATE_PENDING')]
    
    print(f"Total jobs in account: {len(all_jobs)}")
    print(f"Active jobs: {len(active)}")
    
    if len(active) > 0:
        print(f"\n⚠️  WARNING: {len(active)} other jobs are active!")
        print(f"   Your job may be stuck behind these in the queue.")
        for j in active[:3]:
            print(f"   - {j.name}")
    
    print(f"\n{'='*60}")