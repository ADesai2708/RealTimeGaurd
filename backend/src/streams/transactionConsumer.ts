import { redisClient } from "../config/redis";
import {
  predictTransaction,
  MLTransaction,
} from "../services/mlClient";
import { recoverPendingMessages } from "./pendingRecovery";
import { saveTransaction } from "../repositories/transactionRepository";
const TRANSACTION_STREAM = "transactions";
const CONSUMER_GROUP = "fraud-detectors";
const CONSUMER_NAME = "worker-1";

async function processTransaction(
  messageId: string,
  message: Record<string, string>,
): Promise<void> {
  console.log("\nTransaction received:");
  console.log("Message ID:", messageId);
  console.log("Transaction ID:", message.transaction_id);
  console.log("Data:", message);

  const transaction: MLTransaction = {
    transaction_id: message.transaction_id,
    step: Number(message.step),
    type: message.type,
    amount: Number(message.amount),
    oldbalanceOrg: Number(message.oldbalanceOrg),
    newbalanceOrig: Number(message.newbalanceOrig),
    oldbalanceDest: Number(message.oldbalanceDest),
    newbalanceDest: Number(message.newbalanceDest),
  };

  const prediction = await predictTransaction(transaction);

  console.log("\nML Prediction:");
  console.log(
    "Transaction ID:",
    prediction.transaction_id,
  );
  console.log(
    "Fraud Probability:",
    prediction.fraud_probability,
  );
  console.log(
    "Prediction:",
    prediction.prediction,
  );
  console.log(
    "Decision:",
    prediction.decision,
  );
  console.log(
    "ML Latency:",
    `${prediction.latency_ms} ms`,
  );
  console.log(
  "Model Version:",
  prediction.model_version,
);

await saveTransaction({
  transactionId: transaction.transaction_id,

  step: transaction.step,
  type: transaction.type,
  amount: transaction.amount,

  oldBalanceOrg: transaction.oldbalanceOrg,
  newBalanceOrig: transaction.newbalanceOrig,

  oldBalanceDest: transaction.oldbalanceDest,
  newBalanceDest: transaction.newbalanceDest,

  fraudProbability: prediction.fraud_probability,
  prediction: prediction.prediction,
  decision: prediction.decision,

  mlLatencyMs: prediction.latency_ms,
  modelVersion: prediction.model_version,

  createdAt: new Date(),
});

await redisClient.xAck(
  TRANSACTION_STREAM,
  CONSUMER_GROUP,
  messageId,
);

  console.log(
    "Transaction acknowledged:",
    messageId,
  );
}

export async function startTransactionConsumer(): Promise<void> {
  console.log(
    `Starting Redis consumer: ${CONSUMER_NAME}`,
  );

  const recovered =
    await recoverPendingMessages();

  for (const message of recovered) {
    try {
      await processTransaction(
        message.id,
        message.message,
      );
    } catch (error) {
      console.error(
        `Failed to process recovered message ${message.id}:`,
        error,
      );
    }
  }

  while (true) {
    try {
      const result = await redisClient.xReadGroup(
        CONSUMER_GROUP,
        CONSUMER_NAME,
        [
          {
            key: TRANSACTION_STREAM,
            id: ">",
          },
        ],
        {
          COUNT: 1,
          BLOCK: 5000,
        },
      );

      if (!result) {
        continue;
      }

      for (const stream of result) {
        for (const message of stream.messages) {
          try {
            await processTransaction(
              message.id,
              message.message,
            );
          } catch (error) {
            console.error(
              `Failed to process message ${message.id}:`,
              error,
            );
          }
        }
      }
    } catch (error) {
      console.error(
        "Transaction consumer error:",
        error,
      );

      await new Promise((resolve) =>
        setTimeout(resolve, 1000),
      );
    }
  }
}