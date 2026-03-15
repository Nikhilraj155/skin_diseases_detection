import { API_BASE_URL } from "../api/client";
import type { PredictionRecord } from "../types/prediction";

type PredictionCardProps = {
  prediction: PredictionRecord | null;
};

function formatLabel(label: string) {
  return label.replaceAll("_", " ");
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  if (!prediction) {
    return (
      <section className="panel result-panel empty-state">
        <p className="eyebrow">Prediction</p>
        <h2>No result yet</h2>
        <p className="muted">Upload an image to see the analysis results.</p>
      </section>
    );
  }

  return (
    <section className="panel result-panel">
      <p className="eyebrow">Analysis Result</p>
      <h2>{formatLabel(prediction.predicted_label)}</h2>
      <p className="description">{prediction.description}</p>

      <div className="image-frame">
        <img
          src={`${API_BASE_URL.replace(/\/api\/v1$/, "")}${prediction.image_url}`}
          alt={prediction.filename}
        />
      </div>

      <div className="chip-row">
        <span className="chip">{prediction.filename}</span>
      </div>
    </section>
  );
}
