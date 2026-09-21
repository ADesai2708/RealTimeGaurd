import { createClient } from "redis";
import { config } from "./env";

export const redisClient = createClient({
  url: config.redis.url,
});

redisClient.on("error", (error) => {
  console.error("Redis Client Error:", error);
});

export async function connectRedis(): Promise<void> {
  if (!redisClient.isOpen) {
    await redisClient.connect();
  }

  console.log("Redis connected successfully");
}

export async function disconnectRedis(): Promise<void> {
  if (redisClient.isOpen) {
    await redisClient.quit();
  }

  console.log("Redis disconnected");
}