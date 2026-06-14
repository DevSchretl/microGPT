import { useEffect, useRef, useState } from "react";
import SettingsBar from "./components/SettingsBar.jsx";
import MessageList from "./components/MessageList.jsx";
import Composer from "./components/Composer.jsx";
import { streamChat } from "./api.js";

export default function App() {
  const [messages, setMessages] = useState([]); // [{ role, content }]
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(200);
  const [topK, setTopK] = useState(50);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef(null);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((d) => {
        setModels(d.models || []);
        if (d.models?.length) setModel(d.models[0]);
      })
      .catch(() => {});
  }, []);

  // Replace the last (assistant) message via an updater.
  function updateLast(fn) {
    setMessages((cur) => {
      const next = cur.slice();
      next[next.length - 1] = fn(next[next.length - 1]);
      return next;
    });
  }

  async function send(text) {
    const history = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat({
        messages: history,
        model,
        maxNewTokens,
        topK,
        signal: controller.signal,
        onDelta: (delta) => updateLast((m) => ({ ...m, content: m.content + delta })),
      });
    } catch (err) {
      if (err.name !== "AbortError") {
        updateLast((m) => ({ ...m, content: m.content || `[error: ${err.message}]` }));
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function newChat() {
    abortRef.current?.abort();
    setMessages([]);
    setStreaming(false);
  }

  return (
    <div className="app">
      <SettingsBar
        models={models}
        model={model}
        onModel={setModel}
        maxNewTokens={maxNewTokens}
        onMaxNewTokens={setMaxNewTokens}
        topK={topK}
        onTopK={setTopK}
        onNewChat={newChat}
        disabled={streaming}
      />
      <MessageList messages={messages} streaming={streaming} />
      <Composer onSend={send} disabled={streaming} />
    </div>
  );
}
