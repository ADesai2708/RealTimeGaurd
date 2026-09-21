import { randomUUID } from "crypto";
import {
  publishTransaction,
  StreamTransaction,
} from "../streams/transactionStream";

type TransactionScenario =
  | "LEGITIMATE"
  | "SUSPICIOUS"
  | "FRAUD";

function generateLegitimateTransaction(): StreamTransaction {
  const amount = Number(
    (Math.random() * 5000 + 100).toFixed(2),
  );

  const oldbalanceOrg = Number(
    (Math.random() * 50000 + amount).toFixed(2),
  );

  const newbalanceOrig = Number(
    (oldbalanceOrg - amount).toFixed(2),
  );

  const oldbalanceDest = Number(
    (Math.random() * 50000).toFixed(2),
  );

  const newbalanceDest = Number(
    (oldbalanceDest + amount).toFixed(2),
  );

  return {
    transaction_id: `txn-${randomUUID()}`,
    step: Math.floor(Math.random() * 744),
    type: "PAYMENT",
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
  };
}

function generateSuspiciousTransaction(): StreamTransaction {
  const amount = Number(
    (Math.random() * 15000 + 10000).toFixed(2),
  );

  const oldbalanceOrg = Number(
    (amount + Math.random() * 5000).toFixed(2),
  );

  const newbalanceOrig = Number(
    (oldbalanceOrg - amount).toFixed(2),
  );

  const oldbalanceDest = Number(
    Math.random() * 1000,
  );

  const newbalanceDest = Number(
    (oldbalanceDest + amount).toFixed(2),
  );

  return {
    transaction_id: `txn-${randomUUID()}`,
    step: Math.floor(Math.random() * 744),
    type: "TRANSFER",
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
  };
}

function generateFraudTransaction(): StreamTransaction {
  const amount = Number(
    (Math.random() * 50000 + 25000).toFixed(2),
  );

  return {
    transaction_id: `txn-${randomUUID()}`,
    step: Math.floor(Math.random() * 744),
    type: "CASH_OUT",
    amount,
    oldbalanceOrg: amount,
    newbalanceOrig: 0,
    oldbalanceDest: 0,
    newbalanceDest: 0,
  };
}

function generateTransaction(
  scenario: TransactionScenario,
): StreamTransaction {
  switch (scenario) {
    case "LEGITIMATE":
      return generateLegitimateTransaction();

    case "SUSPICIOUS":
      return generateSuspiciousTransaction();

    case "FRAUD":
      return generateFraudTransaction();
  }
}

export async function startTransactionGenerator(
  transactionsPerSecond = 1,
): Promise<void> {
  const scenarios: TransactionScenario[] = [
    "LEGITIMATE",
    "LEGITIMATE",
    "LEGITIMATE",
    "SUSPICIOUS",
    "FRAUD",
  ];

  const intervalMs = 1000 / transactionsPerSecond;

  console.log(
    `Starting transaction generator at ${transactionsPerSecond} transaction(s)/second`,
  );

  while (true) {
    try {
      const scenario =
        scenarios[
          Math.floor(Math.random() * scenarios.length)
        ];

      const transaction =
        generateTransaction(scenario);

      const messageId =
        await publishTransaction(transaction);

      console.log(
        `Generated ${scenario} transaction ${transaction.transaction_id} → ${messageId}`,
      );

      await new Promise((resolve) =>
        setTimeout(resolve, intervalMs),
      );
    } catch (error) {
      console.error(
        "Transaction generator error:",
        error,
      );

      await new Promise((resolve) =>
        setTimeout(resolve, 1000),
      );
    }
  }
}