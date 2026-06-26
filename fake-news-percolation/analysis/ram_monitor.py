"""Run a build_evidence.py subcommand under a memory watchdog (OOM guard).

Samples the process-tree RSS and system memory every 2 s and aborts the child gracefully if
free system memory drops below MIN_AVAIL_GB, so a near-OOM becomes a clean signal rather than a
hard crash or swap thrash. Logs to analysis/_<sub>_mem.log; the child's own stdout/stderr go to
analysis/_<sub>_rerun.log. Exits 0 only if the child finished cleanly and was not aborted.

Any extra arguments are forwarded to build_evidence.py (e.g. --baseline / --influencer):

    python analysis/ram_monitor.py compare
    python analysis/ram_monitor.py influencer --influencer data/my_sweep
"""
import subprocess, sys, os, time, psutil

PY = sys.executable  # run the child with the same interpreter as this watchdog
MIN_AVAIL_GB = 0.3   # abort if free system RAM (GB) falls below this; tune to your machine
HERE = os.path.dirname(os.path.abspath(__file__))

sub = sys.argv[1] if len(sys.argv) > 1 else None
extra = sys.argv[2:]
if sub not in ("eda", "compare", "influencer"):
    sys.exit(f"usage: python {os.path.basename(__file__)} {{eda|compare|influencer}} [build_evidence options]")

total = psutil.virtual_memory().total
log = open(os.path.join(HERE, f"_{sub}_mem.log"), "w")
def emit(s):
    print(s, flush=True); log.write(s + "\n"); log.flush()

emit(f"START sub={sub} total_ram={total/1e9:.1f}GB free_now={psutil.virtual_memory().available/1e9:.1f}GB "
     f"min_avail_abort={MIN_AVAIL_GB}GB")
proc = subprocess.Popen([PY, os.path.join(HERE, "build_evidence.py"), sub, *extra],
                        stdout=open(os.path.join(HERE, f"_{sub}_rerun.log"), "w"), stderr=subprocess.STDOUT)
try:
    p = psutil.Process(proc.pid)
except psutil.NoSuchProcess:
    p = None

peak = 0.0; aborted = False; t0 = time.time()
while proc.poll() is None:
    rss = 0.0
    if p is not None:
        try:
            rss = p.memory_info().rss
            for c in p.children(recursive=True):
                try: rss += c.memory_info().rss
                except psutil.Error: pass
        except psutil.Error:
            rss = 0.0
    vm = psutil.virtual_memory()
    peak = max(peak, rss)
    emit(f"t={time.time()-t0:5.0f}s proc_rss={rss/1e9:5.2f}GB peak={peak/1e9:5.2f}GB "
         f"sys_avail={vm.available/1e9:5.2f}GB sys_used={vm.percent:3.0f}%")
    if vm.available < MIN_AVAIL_GB * 1e9:
        emit("ABORT: system available memory below threshold -> terminating child to avoid OOM")
        proc.terminate()
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired: proc.kill()
        aborted = True
        break
    time.sleep(2)

rc = proc.poll()
emit(f"DONE rc={rc} peak_proc_rss={peak/1e9:.2f}GB elapsed={time.time()-t0:.0f}s aborted={aborted}")
log.close()
sys.exit(0 if (rc == 0 and not aborted) else 1)
