import axios from "axios";
import { config } from "../config/env";

export interface MLTransaction {
  transaction_id: string;
  step: number;
  type: string;
  amount: number;
  oldbalanceOrg: number;
  newbalanceOrig: number;
  oldbalanceDest: number;
  newbalanceDest: number;
}

export interface MLPredictionResponse {
  transaction_id: string;
  fraud_probability: number;
  prediction: "FRAUD" | "LEGITIMATE";
  decision: "APPROVE" | "REVIEW" | "BLOCK";
  latency_ms: number;
  model_version: string;
}

const mlClient = axios.create({
  baseURL: config.mlService.url,
  timeout: 5000,
});

export async function predictTransaction(
  transaction: MLTransaction,
): Promise<MLPredictionResponse> {
  const response = await mlClient.post<MLPredictionResponse>(
    "/predict",
    transaction,
  );

  return response.data;
}