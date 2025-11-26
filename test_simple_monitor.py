"""Simple test of background memory monitoring"""

from luto.tools.mem_monitor import start_memory_monitor, stop_memory_monitor
import time

print("Testing background memory monitoring...")
print("-" * 60)

# Start monitoring
start_memory_monitor()

# Simulate some work that uses memory
print("Creating data...")
data = []
for i in range(2000000):
    data.append(i)
    if i % 500000 == 0:
        print(f"  Progress: {i:,} items")
        time.sleep(0.1)

print("Processing complete!")
print("-" * 60)

# Stop and get results
result = stop_memory_monitor()

if result:
    print("\nDetailed results:")
    print(f"  Duration: {result['duration']:.2f} seconds")
    print(f"  Peak memory delta: {result['peak_memory_mb']:.2f} MB")
    print(f"  Final memory delta: {result['final_memory_mb']:.2f} MB")
    print(f"  Data points collected: {len(result['data'])}")
