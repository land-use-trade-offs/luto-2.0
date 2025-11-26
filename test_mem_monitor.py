"""
Quick test script for the new memory monitor module.
Run this in Jupyter to test the functionality.
"""

from luto.tools.mem_monitor import start_memory_monitor, stop_memory_monitor

# Test 1: Silent monitoring
print("=" * 60)
print("Test 1: Silent background monitoring")
print("=" * 60)

start_memory_monitor()

# Simulate some memory usage
data = []
for i in range(1000000):
    data.append(i)

result = stop_memory_monitor()
print(f"\nResult: {result}")

print("\n" + "=" * 60)
print("Test 2: Live plot monitoring")
print("=" * 60)
print("Run this in Jupyter:")
print("""
from luto.tools.mem_monitor import start_memory_monitor, stop_memory_monitor

# Start with live plot
start_memory_monitor(live=True)

# Your code here
data = [i for i in range(5000000)]

# Stop and see summary
stop_memory_monitor()
""")
