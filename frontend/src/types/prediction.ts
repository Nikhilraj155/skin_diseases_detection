export type PredictionProbability = {
  label: string;
  confidence: number;
};

export type PredictionRecord = {
  id: string;
  filename: string;
  content_type: string;
  predicted_label: string;
  confidence: number;
  description: string;
  fallback_used: boolean;
  probabilities: PredictionProbability[];
  image_url: string;
  created_at: string;
};

export type PredictionHistoryResponse = {
  items: PredictionRecord[];
};
