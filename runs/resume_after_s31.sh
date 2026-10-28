#!/bin/bash
# Resume the large lanes once the §31 TEST caches are built. Watches the ORIGINAL runner (PID 717477) and the real
# cache process only; status-file CRASH lines are ignored (a duplicate runner wrote a false one at 05:37:45).
cd /home/mohhakami/Documents/paper/quworld; S=runs/alloc/status
until { [ -f runs/alloc/cache/active_matter/test.pt ] && [ -f runs/alloc/cache/gray_scott_reaction_diffusion/test.pt ] \
        && ! pgrep -f "[c]ache.py well (active_matter|gray_scott_reaction_diffusion) --test" >/dev/null; } || ! kill -0 717477 2>/dev/null; do sleep 60; done
why=$(kill -0 717477 2>/dev/null && echo "s31 TEST caches built" || echo "s31 runner 717477 exited")
kill -CONT 237509 438458 357694 141818 502187 2>/dev/null; echo "RESUMED §28c §30 §29e §26-K200 §32 runners ($why) $(date -Is)" >> $S
