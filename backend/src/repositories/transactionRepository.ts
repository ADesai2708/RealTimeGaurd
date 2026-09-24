import { getDatabase } from "../config/mongodb";
import { TransactionDocument } from "../models/transaction";

const COLLECTION_NAME = "transactions";

export async function saveTransaction(
  transaction: TransactionDocument,
): Promise<void> {
  const database = getDatabase();

  await database
    .collection<TransactionDocument>(COLLECTION_NAME)
    .insertOne(transaction);

  console.log(
    `Transaction saved to MongoDB: ${transaction.transactionId}`,
  );
}