export default function MessageBubble({ role, content, pending }) {
  return (
    <div className={`bubble-row ${role}`}>
      <div className={`bubble ${role}`}>
        {pending ? <span className="typing">●●●</span> : content}
      </div>
    </div>
  );
}
