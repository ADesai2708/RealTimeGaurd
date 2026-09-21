import { redisClient } from "../config/redis";

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

          await redisClient.xAck(
            TRANSACTION_STREAM,
            CONSUMER_GROUP,
            message.id,
          );

          console.log("Transaction acknowledged:", message.id);
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