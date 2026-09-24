import express from "express";
import { config } from "./config/env";
import {
  connectRedis,
  disconnectRedis,
} from "./config/redis";
import { publishTransaction } from "./streams/transactionStream";
import { createConsumerGroup } from "./streams/consumerGroup";
import { startTransactionConsumer } from "./streams/transactionConsumer";
import { startTransactionGenerator } from "./generator/transactionGenerator";
import { connectMongoDB } from "./config/mongodb";
import { createServer } from "http";
import { initializeSocket } from "./config/socket";

const app = express();
const httpServer = createServer(app);

initializeSocket(httpServer);
app.get("/", (_req, res) => {
  res.json({
    service: "RealTimeGuard Backend",
    status: "running",
  });
});
async function startServer(): Promise<void> {
  try {
    await connectRedis();
await connectMongoDB();
    await createConsumerGroup();

startTransactionConsumer().catch((error) => {
  console.error("Transaction consumer stopped:", error);
});
startTransactionGenerator(1).catch((error) => {
  console.error("Transaction generator stopped:", error);
});

    httpServer.listen(config.port, () => {
  console.log(
    `RealTimeGuard backend running on http://localhost:${config.port}`,
  );
});
  } catch (error) {
    console.error("Failed to start server:", error);
    process.exit(1);
  }
}
app.post("/test/transaction", async (_req, res) => {
  try {
    const messageId = await publishTransaction({
      transaction_id: "test-txn-001",
      step: 120,
      type: "TRANSFER",
      amount: 7500,
      oldbalanceOrg: 15000,
      newbalanceOrig: 7500,
      oldbalanceDest: 1000,
      newbalanceDest: 8500,
    });

    res.json({
      message: "Transaction published",
      stream: "transactions",
      messageId,
    });
  } catch (error) {
    console.error("Failed to publish transaction:", error);

    res.status(500).json({
      error: "Failed to publish transaction",
    });
  }
});

async function shutdown(): Promise<void> {
  console.log("Shutting down server...");

  await disconnectRedis();

  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

startServer();