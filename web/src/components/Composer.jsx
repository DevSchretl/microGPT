import { useState } from "react";

export default function Composer({ onSend, disabled }) {
  const [text, setText] = useState("");

  function submit() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <footer className="composer">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Send a message…  (Enter to send, Shift+Enter for newline)"
        rows={1}
      />
      <button onClick={submit} disabled={disabled || !text.trim()}>
        Send
      </button>
    </footer>
  );
}
