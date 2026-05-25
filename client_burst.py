# client_burst.py
"""
Load testing script for /v1/actions/generations endpoint.
Tests different request rates using Poisson distributions and measures latency.
"""
import base64
import io
import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
import requests


def dummy_image_b64():
    """Generate a dummy 224x224 gray image as base64."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_payload():
    """Create a single action generation payload."""
    return {
        "task": "pick up the red cube",
        "state": [0, 0, 0, 0, 0, 0, 0],
        "images": {
            "image0": dummy_image_b64(),
            "image1": dummy_image_b64(),
            "image2": dummy_image_b64(),
        },
    }


def send_single_request(server_url: str = "http://localhost:8091") -> Tuple[float, int, bool]:
    """
    Send a single request and measure latency.
    Returns: (latency_ms, status_code, success)
    """
    payload = create_payload()
    start = time.perf_counter()
    
    try:
        r = requests.post(
            f"{server_url}/v1/actions/generations",
            json=payload,
            timeout=300,  # 5 min timeout
        )
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        success = r.status_code == 200
        return latency_ms, r.status_code, success
    except Exception as e:
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        print(f"  ❌ Request failed: {e}")
        return latency_ms, -1, False


def test_poisson_burst(
    lambda_param: float,
    num_requests: int = 50,
    server_url: str = "http://localhost:8091",
    max_workers: int = 4,
) -> Dict:
    """
    Test with requests arriving according to Poisson process.
    
    Args:
        lambda_param: Poisson rate parameter (avg requests per second)
        num_requests: Total number of requests to send
        server_url: Server URL
        max_workers: Number of concurrent threads
    
    Returns:
        Dictionary with latency statistics
    """
    print(f"\n📊 Testing with λ={lambda_param:.1f} requests/sec ({num_requests} total requests)")
    print(f"   Expected duration: ~{num_requests/lambda_param:.1f}s")
    
    # Generate inter-arrival times from Poisson process
    # Poisson process: inter-arrival times follow exponential distribution
    inter_arrival_times = np.random.exponential(1.0 / lambda_param, num_requests)
    
    latencies = []
    status_codes = []
    successes = 0
    failures = 0
    
    start_burst = time.perf_counter()
    request_times = []
    
    # Schedule requests according to Poisson times
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for i, inter_arrival in enumerate(inter_arrival_times):
            # Calculate when this request should be sent
            scheduled_time = start_burst + sum(inter_arrival_times[:i+1])
            current_time = time.perf_counter()
            wait_time = max(0, scheduled_time - current_time)
            
            # Submit request (will start immediately or after wait_time)
            if wait_time > 0:
                time.sleep(min(wait_time, 0.1))  # Sleep in small increments
            
            future = executor.submit(send_single_request, server_url)
            futures[future] = i
            request_times.append(current_time)
        
        # Collect results
        for future in as_completed(futures):
            req_num = futures[future]
            latency_ms, status_code, success = future.result()
            latencies.append(latency_ms)
            status_codes.append(status_code)
            
            if success:
                successes += 1
            else:
                failures += 1
            
            if (req_num + 1) % 10 == 0:
                print(f"   ✓ Completed {req_num + 1}/{num_requests} requests")
    
    end_burst = time.perf_counter()
    actual_duration = end_burst - start_burst
    
    # Calculate statistics
    if latencies:
        stats = {
            "lambda": lambda_param,
            "num_requests": num_requests,
            "successes": successes,
            "failures": failures,
            "success_rate": (successes / num_requests * 100) if num_requests > 0 else 0,
            "actual_duration_sec": actual_duration,
            "actual_throughput": num_requests / actual_duration if actual_duration > 0 else 0,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "stdev_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "latencies": latencies,
        }
    else:
        stats = {"lambda": lambda_param, "error": "No successful requests"}
    
    return stats


def print_results(all_stats: List[Dict]):
    """Pretty-print results from all load tests."""
    print("\n" + "="*100)
    print("📈 LOAD TEST RESULTS SUMMARY")
    print("="*100)
    
    print(f"\n{'λ (req/s)':>15} | {'Requests':>10} | {'Success %':>10} | "
          f"{'Actual λ':>10} | {'Mean (ms)':>10} | {'Median (ms)':>10} | "
          f"{'P95 (ms)':>10} | {'P99 (ms)':>10}")
    print("-" * 110)
    
    for stats in all_stats:
        if "error" in stats:
            lambda_str = str(stats['lambda'])
            print(f"{lambda_str:>15} | {'ERROR':>10}")
            continue
        
        lambda_str = str(stats['lambda']) if isinstance(stats['lambda'], str) else f"{stats['lambda']:.1f}"
        
        print(
            f"{lambda_str:>15} | "
            f"{stats['num_requests']:>10} | "
            f"{stats['success_rate']:>9.1f}% | "
            f"{stats.get('actual_throughput', 0):>10.2f} | "
            f"{stats['mean_latency_ms']:>10.2f} | "
            f"{stats['median_latency_ms']:>10.2f} | "
            f"{stats['p95_latency_ms']:>10.2f} | "
            f"{stats['p99_latency_ms']:>10.2f}"
        )
    
    print("-" * 110)
    
    # Find inflection point where latency starts degrading
    print("\n📊 Key Observations:")
    numeric_stats = [s for s in all_stats[1:] if "error" not in s]  # Skip baseline
    if len(numeric_stats) > 1:
        means = [s["mean_latency_ms"] for s in numeric_stats]
        if len(means) > 1:
            for i in range(1, len(means)):
                if means[i] > means[i-1] * 1.5:  # 50% increase
                    lambda_val = numeric_stats[i]['lambda']
                    print(f"   ⚠️  Latency degradation noticed at λ={lambda_val:.1f}")
                    break
    
    # Print latency breakdown for highest load
    load_stats = [s for s in all_stats if "error" not in s and isinstance(s.get('lambda'), (int, float))]
    if load_stats:
        highest_load = max(load_stats, key=lambda x: x["lambda"])
        print(f"\n   Highest Load Test (λ={highest_load['lambda']:.1f}):")
        print(f"     • Mean latency: {highest_load['mean_latency_ms']:.2f}ms")
        print(f"     • Std deviation: {highest_load['stdev_latency_ms']:.2f}ms")
        print(f"     • Range: {highest_load['min_latency_ms']:.2f}ms - {highest_load['max_latency_ms']:.2f}ms")
        print(f"     • Duration: {highest_load['actual_duration_sec']:.2f}s")


def test_sequential_baseline(server_url: str = "http://localhost:8091", num_requests: int = 10) -> Dict:
    """
    Test requests sent sequentially (one at a time) to measure baseline latency.
    """
    print(f"\n⏱️  Baseline: Sequential Requests ({num_requests} requests)")
    
    latencies = []
    successes = 0
    
    for i in range(num_requests):
        latency_ms, status_code, success = send_single_request(server_url)
        latencies.append(latency_ms)
        if success:
            successes += 1
        
        if (i + 1) % 5 == 0:
            print(f"   ✓ {i + 1}/{num_requests} completed")
    
    stats = {
        "lambda": "sequential",
        "num_requests": num_requests,
        "successes": successes,
        "failures": num_requests - successes,
        "success_rate": (successes / num_requests * 100) if num_requests > 0 else 0,
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "stdev_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
    }
    
    return stats


def main():
    server_url = "http://localhost:8091"
    
    print("🚀 vLLM-Omni Action Generation Load Test")
    print(f"   Target: {server_url}/v1/actions/generations")
    print("   Testing with Poisson-distributed request arrivals\n")
    
    # FIRST: Get baseline with sequential requests
    baseline_stats = test_sequential_baseline(server_url, num_requests=10)
    
    # Test different Poisson rates (lambda values)
    # Start with low load, gradually increase
    lambda_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    num_requests_per_test = 30
    
    all_stats = [baseline_stats]
    
    for lambda_param in lambda_values:
        try:
            stats = test_poisson_burst(
                lambda_param=lambda_param,
                num_requests=num_requests_per_test,
                server_url=server_url,
                max_workers=max(1, int(lambda_param) + 1),
            )
            all_stats.append(stats)
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test failed for λ={lambda_param}: {e}")
    
    # Print summary
    print_results(all_stats)
    
    # Analyze queueing
    print("\n🔍 QUEUEING ANALYSIS:")
    baseline_mean = baseline_stats["mean_latency_ms"]
    print(f"   Baseline (sequential): {baseline_mean:.2f}ms per request")
    
    for stats in all_stats[1:]:  # Skip baseline
        if "error" not in stats:
            slowdown = stats["mean_latency_ms"] / baseline_mean
            lambda_str = f"{stats['lambda']:.1f}" if isinstance(stats['lambda'], (int, float)) else str(stats['lambda'])
            print(f"   λ={lambda_str}: {stats['mean_latency_ms']:.2f}ms ({slowdown:.1f}x baseline)")
            if slowdown > 3:
                print(f"      ⚠️  SEVERE queueing detected! Requests are waiting {slowdown:.1f}x longer")
    
    # Save results to JSON
    results_file = "load_test_results.json"
    with open(results_file, "w") as f:
        # Convert to JSON-serializable format
        json_stats = []
        for s in all_stats:
            s_copy = s.copy()
            s_copy.pop("latencies", None)  # Remove raw latency list
            json_stats.append(s_copy)
        json.dump(json_stats, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")


if __name__ == "__main__":
    main()
