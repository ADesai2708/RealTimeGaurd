import { MongoClient, Db } from "mongodb";
import { config } from "./env";

const mongoClient = new MongoClient(
  config.mongodb.uri,
);

let database: Db | null = null;

export async function connectMongoDB(): Promise<void> {
  if (database) {
    return;
  }

  await mongoClient.connect();

  database = mongoClient.db(
    config.mongodb.database,
  );

  await database.command({ ping: 1 });

  console.log(
    `MongoDB connected successfully: ${config.mongodb.database}`,
  );
}

export function getDatabase(): Db {
  if (!database) {
    throw new Error(
      "MongoDB is not connected. Call connectMongoDB() first.",
    );
  }

  return database;
}

export async function disconnectMongoDB(): Promise<void> {
  if (!database) {
    return;
  }

  await mongoClient.close();

  database = null;

  console.log("MongoDB disconnected");
}