#!/bin/bash
# Deliver a bridge message to the GPT Codex session (tmux codexqu) only after the user's own text has left its input box
# (submitted: it then shows in the history with the history background, not the composer's). Gives up after 3 h.
cd /home/mohhakami/Documents/paper/quworld; MARK="$1"; SUBJ="$2"; BODY="$3"
for i in $(seq 1 1080); do
  if tmux capture-pane -t codexqu -p -e -S -80 | grep -F "$MARK" | grep -q "48;2;60;60;60"; then
    sleep 5; echo "codexqu:0.0" > .bridge/sol_target
    BRIDGE_FROM=claude .bridge/send.sh sol "$SUBJ" < "$BODY"; echo "DELIVERED $(date -Is)"; exit 0
  fi
  sleep 10
done
echo "NOT_DELIVERED (user text still in the composer after 3 h)"; exit 1
