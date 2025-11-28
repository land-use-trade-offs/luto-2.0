# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J.,
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S.,
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.


"""
Memory monitoring utilities for tracking memory usage with live Plotly visualization.
"""

import gc
import os
import threading
import time
import functools

import pandas as pd
import psutil
import plotly.graph_objects as go
from IPython.display import display, clear_output


# Global state
memory_log = []
monitoring = False
monitor_thread = None
live_plot_thread = None
baseline_memory = 0
stop_live_plot = threading.Event()


def monitor_memory(interval=0.01):
    """
    Memory monitoring focused on Working Set delta from baseline.
    Runs in a background thread, logs memory usage every `interval` seconds.
    Includes child processes (multiprocessing support).
    """
    process = psutil.Process(os.getpid())

    while monitoring:
        try:
            memory_info = process.memory_info()

            # Check if working set is available and use consistently
            has_wset = hasattr(memory_info, 'wset')

            if has_wset:
                current_wset_mb = memory_info.wset / 1024 ** 2
            else:
                current_wset_mb = memory_info.rss / 1024 ** 2

            # Include child processes using the SAME metric type
            children = process.children(recursive=True)
            if children:
                for child in children:
                    try:
                        child_memory_info = child.memory_info()
                        if has_wset and hasattr(child_memory_info, 'wset'):
                            current_wset_mb += child_memory_info.wset / 1024 ** 2
                        else:
                            # Use RSS for consistency if wset not available
                            current_wset_mb += child_memory_info.rss / 1024 ** 2
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

            # Calculate delta from baseline
            delta_mb = current_wset_mb - baseline_memory

            # Store delta memory info
            memory_log.append({
                'time': time.time(),
                'wset_mb': current_wset_mb,
                'delta_mb': delta_mb
            })

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

        time.sleep(interval)


def _show_live_plot(update_interval=0.1):
    """
    Internal function to display live-updating plot in background thread.
    Runs until stop_live_plot event is set.
    """
    print("Live plot active in background. Run your code normally.")
    print("The plot will update automatically.")
    print("Call stop_memory_monitor() when done.")
    print("-" * 60)

    try:
        while not stop_live_plot.is_set() and monitoring:
            if memory_log:
                # Make thread-safe copy to avoid race condition
                log_copy = memory_log.copy()

                # Convert to DataFrame
                df = pd.DataFrame(log_copy)
                df['Time'] = df['time'] - df['time'].min()

                # Create plot
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df['Time'],
                    y=df['delta_mb'],
                    mode='lines',
                    name='Delta Memory',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='<b>Time</b>: %{x:.2f}s<br><b>Memory</b>: %{y:.2f} MB<extra></extra>'
                ))

                fig.update_layout(
                    title='Live Memory Usage (Delta from baseline)',
                    xaxis_title='Time (s)',
                    yaxis_title='Delta Memory (MB)',
                    hovermode='x unified',
                    template='plotly_white',
                    showlegend=True,
                    height=400,
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
                )

                # Clear previous output and display new plot
                clear_output(wait=True)
                display(fig)

            time.sleep(update_interval)

    except KeyboardInterrupt:
        print("\nLive monitoring interrupted by user.")

    # Show final plot
    if memory_log:
        # Make thread-safe copy to avoid race condition
        log_copy = memory_log.copy()

        df = pd.DataFrame(log_copy)
        df['Time'] = df['time'] - df['time'].min()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df['delta_mb'],
            mode='lines',
            name='Delta Memory',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Time</b>: %{x:.2f}s<br><b>Memory</b>: %{y:.2f} MB<extra></extra>'
        ))

        fig.update_layout(
            title='Final Memory Usage Delta',
            xaxis_title='Time (s)',
            yaxis_title='Delta Memory (MB)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            height=400,
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        )

        clear_output(wait=True)
        display(fig)


def start_memory_monitor(update_interval=0.1, enable_live_plot=True):
    """
    Start Working Set memory monitoring with baseline measurement.
    Monitoring runs in background thread with optional live plot updates.

    Includes child processes (multiprocessing support) - tracks memory usage
    across all spawned child processes recursively.

    Parameters:
        update_interval (float): Seconds between plot updates. Default: 0.1
        enable_live_plot (bool): If True, shows live updating plot. If False, only shows final plot. Default: True

    Usage:
        from luto.tools.mem_monitor import start_memory_monitor, stop_memory_monitor

        start_memory_monitor()  # Starts monitoring with live plot
        my_code()  # Your code runs while plot updates in background
        stop_memory_monitor()  # Stops and shows final summary
    """
    global monitoring, monitor_thread, live_plot_thread, baseline_memory, stop_live_plot

    # Stop any existing monitoring first
    if monitoring or (monitor_thread and monitor_thread.is_alive()):
        print("Stopping existing monitor...")
        monitoring = False
        stop_live_plot.set()

        # Wait for old threads to finish
        if monitor_thread and monitor_thread.is_alive():
            monitor_thread.join(timeout=1)
        if live_plot_thread and live_plot_thread.is_alive():
            live_plot_thread.join(timeout=2)

    # Clear previous log and reset state
    memory_log.clear()
    stop_live_plot.clear()
    monitor_thread = None
    live_plot_thread = None

    # Force garbage collection to get clean baseline
    gc.collect()
    time.sleep(0.1)  # Give GC time to complete

    # Get baseline memory usage (including child processes)
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    has_wset = hasattr(memory_info, 'wset')
    if has_wset:
        baseline_memory = memory_info.wset / 1024 ** 2
    else:
        baseline_memory = memory_info.rss / 1024 ** 2

    # Include child processes in baseline using the SAME metric type
    children = process.children(recursive=True)
    if children:
        for child in children:
            try:
                child_memory_info = child.memory_info()
                if has_wset and hasattr(child_memory_info, 'wset'):
                    baseline_memory += child_memory_info.wset / 1024 ** 2
                else:
                    # Use RSS for consistency if wset not available
                    baseline_memory += child_memory_info.rss / 1024 ** 2
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    # Start monitoring in background thread
    monitoring = True
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    print(f"Memory monitoring started (baseline: {baseline_memory:.2f} MB, including child processes)")

    # Start plot updates in another background thread (if enabled)
    if enable_live_plot:
        try:
            live_plot_thread = threading.Thread(target=_show_live_plot, args=(update_interval,), daemon=True)
            live_plot_thread.start()
            # Give the live plot thread a moment to initialize
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Could not start live plot (requires Jupyter): {e}")
            print("Continuing with silent monitoring...")
    else:
        print("Live plot disabled. Print statements will be visible. Final plot will be shown when monitoring stops.")


def stop_memory_monitor(return_data=False):
    """
    Stop memory monitoring and show summary.

    Parameters:
        return_data (bool): If True, returns dictionary with full data. Default: False

    Returns:
        dict or None: If return_data=True, returns statistics dict. Otherwise None.
    """
    global monitoring, stop_live_plot

    # Stop both threads
    monitoring = False
    stop_live_plot.set()

    # Wait for monitor thread to finish
    if monitor_thread:
        monitor_thread.join(timeout=1)

    # Wait for live plot thread to finish (this will show final plot)
    if live_plot_thread:
        live_plot_thread.join(timeout=2)

    if not memory_log:
        print("No memory data collected")
        return None

    # Make thread-safe copy to avoid race condition
    log_copy = memory_log.copy()

    # Convert to DataFrame
    df = pd.DataFrame(log_copy)
    df['Time'] = df['time'] - df['time'].min()

    # Calculate summary statistics
    max_mem = df['delta_mb'].max()
    final_mem = df['delta_mb'].iloc[-1]
    duration = df['Time'].iloc[-1]

    # Print summary (live plot already showed the graph)
    print(f"\nMonitoring stopped.")
    print(f"Duration: {duration:.2f}s | Peak: {max_mem:.2f} MB | Final: {final_mem:.2f} MB")

    # Only return data if explicitly requested
    if return_data:
        return {
            'duration': duration,
            'peak_memory_mb': max_mem,
            'final_memory_mb': final_mem,
            'data': df
        }

    return None


def trace_mem_usage(func=None, *, update_interval=0.1, return_data=False, live_plot=True):
    """
    Trace memory usage of a function with automatic lifecycle management.

    Can be used as a decorator or as a regular function wrapper.
    This is the recommended way to monitor memory usage. It handles starting,
    monitoring, and cleanup automatically, even if the function raises an error.

    Includes child processes (multiprocessing support) - tracks memory usage
    across all spawned child processes recursively.

    Parameters:
        func (callable): The function to monitor (when used as function wrapper)
        update_interval (float): Seconds between plot updates. Default: 0.1
        return_data (bool): If True, returns both function result and memory stats. Default: False
        live_plot (bool): If True, enables live plotting (will suppress print statements). Default: True

    Returns:
        When used as decorator: Returns wrapped function
        When used as wrapper: Returns function result (or tuple with stats if return_data=True)


    Examples:
        # Default: Live plot enabled (print statements suppressed)
        @trace_mem_usage
        def expensive_operation(size):
            data = [i**2 for i in range(size)]
            return sum(data)

        result = expensive_operation(1000000)
    """
    

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            print(f"Starting memory trace for: {fn.__name__}")
            print("-" * 60)

            # Start monitoring (live plot enabled by default)
            start_memory_monitor(update_interval=update_interval, enable_live_plot=live_plot)

            function_result = None
            error_occurred = False

            try:
                # Execute the function
                function_result = fn(*args, **kwargs)

            except Exception as e:
                error_occurred = True
                print(f"\n{'='*60}")
                print(f"ERROR: Function '{fn.__name__}' raised an exception:")
                print(f"{type(e).__name__}: {e}")
                print(f"{'='*60}")
                raise  # Re-raise the exception after stopping monitor

            finally:
                # Always stop monitoring, even if function fails
                memory_stats = stop_memory_monitor(return_data=True)

                if error_occurred:
                    print("\nMemory monitoring stopped due to error.")
                else:
                    print(f"\nFunction '{fn.__name__}' completed successfully.")

            # Return results
            if return_data and memory_stats:
                return function_result, memory_stats
            else:
                return function_result

        return wrapper

    return decorator(func)
