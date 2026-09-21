import { redisClient } from "../config/redis";

const TRANSACTION_STREAM = "transactions";

export interface StreamTransaction {
  transaction_id: string;
  step: number;
  type: string;
  amount: number;
  oldbalanceOrg: number;
  newbalanceOrig: number;
  oldbalanceDest: number;
  newbalanceDest: number;
}

export async function publishTransaction(
  transaction: StreamTransaction,
): Promise<string> {
  const messageId = await redisClient.xAdd(
    TRANSACTION_STREAM,
    "*",
    {
      transaction_id: transaction.transaction_id,
      step: transaction.step.toString(),
      type: transaction.type,
      amount: transaction.amount.toString(),
      oldbalanceOrg: transaction.oldbalanceOrg.toString(),
      newbalanceOrig: transaction.newbalanceOrig.toString(),
      oldbalanceDest: transaction.oldbalanceDest.toString(),
      newbalanceDest: transaction.newbalanceDest.toString(),
    },
  );

  return messageId;
}