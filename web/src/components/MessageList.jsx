import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble.jsx";

export default function MessageList({ messages, streaming }) {
  const endRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <main className="messages">
      {messages.length === 0 && <div className="empty">Ask me anything…</div>}
      {messages.map((m, i) => (
        <MessageBubble
          key={i}
          role={m.role}
          content={m.content}
          pending={
            streaming &&
            i === messages.length - 1 &&
            m.role === "assistant" &&
            m.content === ""
          }
        />
      ))}
      <div ref={endRef} />
    </main>
  );
}
