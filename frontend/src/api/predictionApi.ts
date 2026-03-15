import { apiRequest } from "./client";
import type { PredictionHistoryResponse, PredictionRecord } from "../types/prediction";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export async function createPrediction(file: File): Promise<PredictionRecord> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predictions`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Upload failed.");
  }

  return response.json() as Promise<PredictionRecord>;
}

export function getHistory(): Promise<PredictionHistoryResponse> {
  return apiRequest<PredictionHistoryResponse>("/history");
}

// New: Get detailed disease report
export type DiseaseReport = {
  disease_name: string;
  description: string;
  symptoms: string[];
  risk_factors: string[];
  recommendations: string[];
  when_to_see_doctor: string;
  prediction_id: string;
  filename: string;
  confidence: number;
  confidence_percentage: number;
  image_url: string;
  created_at: string;
  generated_by: string;
};

export async function getDiseaseReport(predictionId: string): Promise<DiseaseReport> {
  const response = await fetch(`${API_BASE_URL}/predictions/${predictionId}/report`);
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Failed to get report.");
  }

  return response.json() as Promise<DiseaseReport>;
}
