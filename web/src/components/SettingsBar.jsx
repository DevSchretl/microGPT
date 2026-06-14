export default function SettingsBar({
  models,
  model,
  onModel,
  maxNewTokens,
  onMaxNewTokens,
  topK,
  onTopK,
  onNewChat,
  disabled,
}) {
  return (
    <header className="settings">
      <div className="brand">GPT Chat</div>
      <div className="controls">
        <label>
          Model
          <select value={model} onChange={(e) => onModel(e.target.value)} disabled={disabled}>
            {models.length === 0 && <option value="">(none)</option>}
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label>
          Max tokens
          <input
            type="number"
            min="1"
            max="1000"
            value={maxNewTokens}
            onChange={(e) => onMaxNewTokens(Number(e.target.value))}
            disabled={disabled}
          />
        </label>
        <label>
          Top-k
          <input
            type="number"
            min="1"
            max="200"
            value={topK}
            onChange={(e) => onTopK(Number(e.target.value))}
            disabled={disabled}
          />
        </label>
        <button className="newchat" onClick={onNewChat}>
          New chat
        </button>
      </div>
    </header>
  );
}
