export interface TransactionDocument {
  transactionId: string;

  step: number;
  type: string;
  amount: number;

  oldBalanceOrg: number;
  newBalanceOrig: number;

  oldBalanceDest: number;
  newBalanceDest: number;

  fraudProbability: number;
  prediction: "FRAUD" | "LEGITIMATE";
  decision: "APPROVE" | "REVIEW" | "BLOCK";

  mlLatencyMs: number;
  modelVersion: string;

  createdAt: Date;
}