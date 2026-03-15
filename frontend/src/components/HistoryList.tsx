import type { PredictionRecord } from "../types/prediction";

type HistoryListProps = {
  items: PredictionRecord[];
  onSelect: (item: PredictionRecord) => void;
};

function formatLabel(label: string) {
  return label.replaceAll("_", " ");
}

export function HistoryList({ items, onSelect }: HistoryListProps) {
  return (
    <section className="panel history-panel">
      <div className="history-header">
        <div>
          <p className="eyebrow">History</p>
          <h2>Recent analyses</h2>
        </div>
      </div>

      <div className="history-list">
        {items.length === 0 ? (
          <p className="muted">No stored predictions yet.</p>
        ) : (
          items.map((item) => (
            <button className="history-item" key={item.id} type="button" onClick={() => onSelect(item)}>
              <span>{formatLabel(item.predicted_label)}</span>
            </button>
          ))
        )}
      </div>
    </section>
  );
}
