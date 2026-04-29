interface ToolEvent {
  name: string;
  type: 'start' | 'end';
  output?: string;
}

const ToolTimeline = ({ events }: { events: ToolEvent[] }) => {
  if (!events.length) return null;
  return (
    <div style={{ marginTop: 20, padding: 10, backgroundColor: '#f0f4f8', borderRadius: 8 }}>
      <h4>Tools used</h4>
      {events.map((evt, i) => (
        <div key={i} style={{ marginBottom: 8 }}>
          {evt.type === 'start' && (
            <p><strong>⚙️ Calling {evt.name}</strong></p>
          )}
          {evt.type === 'end' && (
            <p style={{ paddingLeft: 20 }}>✅ {evt.name} completed: {evt.output?.slice(0, 100)}</p>
          )}
        </div>
      ))}
    </div>
  );
};

export default ToolTimeline;