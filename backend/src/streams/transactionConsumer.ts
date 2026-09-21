import { redisClient } from "../config/redis";
import {
  predictTransaction,
  MLTransaction,
} from "../services/mlClient";

const TRANSACTION_STREAM = "transactions";
const CONSUMER_GROUP = "fraud-detectors";
const CONSUMER_NAME = "worker-1";

export async function startTransactionConsumer(): Promise<void> {
  console.log(
    `Starting Redis consumer: ${CONSUMER_NAME}`,
  );

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
          console.log("\nTransaction received:");
          console.log("Message ID:", message.id);
          console.log("Data:", message.message);

          const transaction: MLTransaction = {
            transaction_id: message.message.transaction_id,
            step: Number(message.message.step),
            type: message.message.type,
            amount: Number(message.message.amount),
            oldbalanceOrg: Number(
              message.message.oldbalanceOrg,
            ),
            newbalanceOrig: Number(
              message.message.newbalanceOrig,
            ),
            oldbalanceDest: Number(
              message.message.oldbalanceDest,
            ),
            newbalanceDest: Number(
              message.message.newbalanceDest,
            ),
          };

          const prediction =
            await predictTransaction(transaction);

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

          await redisClient.xAck(
            TRANSACTION_STREAM,
            CONSUMER_GROUP,
            message.id,
          );

          console.log(
            "Transaction acknowledged:",
            message.id,
          );
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