// POST the conversation and stream the assistant reply via Server-Sent Events.
// Calls onDelta(text) for each chunk; resolves when the stream signals done.
export async function streamChat({ messages, model, maxNewTokens, topK, signal, onDelta }) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      model,
      max_new_tokens: maxNewTokens,
      top_k: topK,
    }),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`request failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line.
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const event = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of event.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice(5).trim());
        if (payload.error) throw new Error(payload.error);
        if (payload.delta) onDelta(payload.delta);
        if (payload.done) return;
      }
    }
  }
}
