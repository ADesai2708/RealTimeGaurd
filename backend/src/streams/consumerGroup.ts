import { redisClient } from "../config/redis";

const TRANSACTION_STREAM = "transactions";
const CONSUMER_GROUP = "fraud-detectors";

export async function createConsumerGroup(): Promise<void> {
  try {
    await redisClient.xGroupCreate(
      TRANSACTION_STREAM,
      CONSUMER_GROUP,
      "0",
      {
        MKSTREAM: true,
      },
    );

    console.log(
      `Redis consumer group "${CONSUMER_GROUP}" created`,
    );
  } catch (error) {
    if (
      error instanceof Error &&
      error.message.includes("BUSYGROUP")
    ) {
      console.log(
        `Redis consumer group "${CONSUMER_GROUP}" already exists`,
      );

      return;
    }

    throw error;
  }
}