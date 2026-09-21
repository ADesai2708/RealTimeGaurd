import { redisClient } from "../config/redis";

const TRANSACTION_STREAM = "transactions";
const CONSUMER_GROUP = "fraud-detectors";
const CONSUMER_NAME = "worker-1";

const MIN_IDLE_TIME_MS = 5_000;

export interface RecoveredMessage {
  id: string;
  message: Record<string, string>;
}

export async function recoverPendingMessages(): Promise<
  RecoveredMessage[]
> {
  try {
    const result = await redisClient.xAutoClaim(
      TRANSACTION_STREAM,
      CONSUMER_GROUP,
      CONSUMER_NAME,
      MIN_IDLE_TIME_MS,
      "0-0",
      {
        COUNT: 10,
      },
    );

    if (result.messages.length === 0) {
      return [];
    }

    console.log(
      `Recovered ${result.messages.length} pending transaction(s)`,
    );

    return result.messages.reduce<RecoveredMessage[]>(
      (recoveredMessages, message) => {
        if (message !== null) {
          recoveredMessages.push({
            id: message.id,
            message: message.message,
          });
        }

        return recoveredMessages;
      },
      [],
    );
  } catch (error) {
    console.error(
      "Pending message recovery error:",
      error,
    );

    return [];
  }
}